# GPU Pulse / Colab single cell. 60-second real CUPTI capture + GEMM throughput.
import os, sys, site, json, pathlib, subprocess, importlib.util, tempfile, html
from importlib.metadata import version, PackageNotFoundError

def install_if_missing(module, package):
    if importlib.util.find_spec(module) is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

for module, package in [('cupti','cupti-python==13.5.0'),('torch','torch'),('pandas','pandas'),('plotly','plotly')]:
    install_if_missing(module, package)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, HTML

try:
    print('CUPTI Python:', version('cupti-python'))
    print('CUPTI C package:', version('nvidia-cuda-cupti'))
except PackageNotFoundError:
    print('CUDA 13 CUPTI C package not detected.')

# Six actual workload phases; all curves use the same 1-second time axis.
# Change CAPTURE_SECONDS only together with phase boundaries below.
CAPTURE_SECONDS = 60
SAMPLE_INTERVAL_NS = 1_000_000_000  # GPU_TIME_INTERVAL: 1 second in nanoseconds
REPORT_INTERVAL_SECONDS = 1.0  # Common cadence for CUPTI, GEMM and charts
MAX_SAMPLES = 100_000

# Probe in clean child processes: loading a different .so in this notebook process is unsafe.
paths = []
for base in [pathlib.Path(p) for p in site.getsitepackages()] + list(pathlib.Path('/usr/local').glob('cuda*')):
    for glob in ['nvidia/cu13/lib/libcupti.so*', 'extras/CUPTI/lib64/libcupti.so*',
                 'targets/x86_64-linux/lib/libcupti.so*']:
        paths += list(base.glob(glob))
paths = list(dict.fromkeys(str(p.resolve()) for p in paths if p.is_file()))
probe = "import ctypes,sys; x=ctypes.CDLL(sys.argv[1]); print(int(hasattr(x,'cuptiProfilerHostGetSinglePassSets')))"
valid = []
for path in paths:
    r = subprocess.run([sys.executable,'-c',probe,path], capture_output=True,text=True)
    if r.returncode == 0 and r.stdout.strip() == '1':
        valid.append(path)
print('CUPTI libraries tested:', len(paths), '| valid single-pass API:', len(valid))
if not valid:
    raise RuntimeError('CUPTI 13.x 라이브러리에 필수 Single-pass API가 없습니다. '
                       '라이브러리 경로/설치를 수정해야 하며, 메트릭을 임의 생성하지 않습니다. '
                       'Candidates: '+repr(paths[:8]))

lib_path = valid[0]
folder = pathlib.Path(tempfile.mkdtemp(prefix='gpupulse_'))
output_path = folder / 'output.json'

# Isolated process: preload selected native CUPTI library before cupti bindings.
CHILD = r'''
import os, json, ctypes, math, time
import torch
lib = ctypes.CDLL(os.environ['GPU_PULSE_LIB'],mode=ctypes.RTLD_GLOBAL)
assert hasattr(lib,'cuptiProfilerHostGetSinglePassSets')
from cupti import pm_sampling, profiler_host

CAPTURE_SECONDS = float(os.environ['GPU_PULSE_CAPTURE_SECONDS'])
SAMPLE_INTERVAL_NS = int(os.environ['GPU_PULSE_SAMPLE_INTERVAL_NS'])
REPORT_INTERVAL_SECONDS = float(os.environ['GPU_PULSE_REPORT_INTERVAL_SECONDS'])
MAX_SAMPLES = int(os.environ['GPU_PULSE_MAX_SAMPLES'])
torch.cuda.set_device(0)
torch.empty(1,device='cuda')
torch.cuda.synchronize()
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print('GPU:',name,'| compute capability:',cap,flush=True)

supported = profiler_host.get_supported_chips()
proposals = []
upper = name.upper()
if 'B300' in upper or 'GB300' in upper: proposals=['GB300','GB202','GB200','GB100']
elif 'B200' in upper or 'GB200' in upper: proposals=['GB100','GB200','GB202']
elif 'H200' in upper or 'H100' in upper: proposals=['GH100']
elif 'A100' in upper or 'A800' in upper: proposals=['GA100']
elif 'L40S' in upper or 'L40' in upper: proposals=['AD102']
elif 'L4' in upper: proposals=['AD104']
chip = next((x for x in proposals if x in supported),None)
if chip is None:
    # Unknown GPU: match supported chip prefix to the architecture.
    architecture = {8:['GA','GA10'],9:['GH','AD'],10:['GB'],12:['GB']}.get(cap[0],[])
    chip = next((x for x in supported if any(x.startswith(a) for a in architecture)),None)
if chip is None:
    raise RuntimeError('Supported chip name could not be resolved. Choices: '+repr(supported))
print('CUPTI chip:',chip,flush=True)
sets = list(profiler_host.get_single_pass_sets(chip))
print('Single-pass sets:',sets,flush=True)

# NVIDIA uses distinct metric naming across architectures/sets.
# The following are candidates, NOT guaranteed PM Sampling capabilities.
BASE = {
 'gr':'gr__cycles_elapsed.max',
 'sm':'sm__cycles_elapsed.sum',
 'read':'dramc__sectors_op_read.sum',
 'write':'dramc__sectors_op_write.sum',
 'l1':'l1tex__t_sector_hit_rate.pct',
 'l2':'lts__average_t_sector_hit_rate_realtime.pct',
 'fadd':'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum',
 'fmul':'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum',
 'ffma':'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum',
 'tensor_active':'sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed',
 'tensor_inst':'smsp__inst_executed_pipe_tensor.sum',
 'fma_inst':'smsp__inst_executed_pipe_fma.sum',
}

# Enumerate metrics actually exposed by each set; never trust GPU model alone.
catalog = {}
for s in sets:
    try:
        catalog[s] = list(profiler_host.get_metrics_in_single_pass_set(chip,s))
        print('Set',s,'catalog entries:',len(catalog[s]),flush=True)
    except Exception as e:
        print('Catalog failed',s,repr(e),flush=True)

# Probing individual settings determines which metrics this runtime accepts.
# First prefer the successful six-counter baseline, then test additions.
priority_sets = [s for s in ['TriageSCG','TriageAC'] if s in sets]
priority_sets += [s for s in sets if s not in priority_sets]
priority_sets += [None]

errors = []
def configure(keys, set_name):
    c = pm_sampling.Collector(device_index=0)
    try:
        c.enable()
        kw = dict(metrics=[BASE[k] for k in keys],
                  hardware_buffer_size=256*1024*1024,
                  sampling_interval=SAMPLE_INTERVAL_NS,
                  trigger_mode=pm_sampling.TriggerMode.GPU_TIME_INTERVAL)
        if set_name is not None: kw['single_pass_metric_set_name']=set_name
        c.configure(**kw)
        return c
    except Exception as e:
        errors.append((str(set_name),list(keys),str(e)[:140]))
        try: c.disable()
        except Exception: pass
        return None

main = ['gr','sm','read','write','l1','l2']
chosen_set=None; chosen=[]; collector=None
for s in priority_sets:
    c=configure(main,s)
    if c is not None:
        collector=c; chosen=list(main); chosen_set=s; break
if collector is None:
    # Keep the best successful collector alive; close other candidates.
    best_collector = None
    best_keys = []
    best_set = None
    for s in priority_sets:
        trial_keys = []
        trial_collector = None
        for k in main:
            candidate = configure(trial_keys + [k], s)
            if candidate is not None:
                if trial_collector is not None:
                    trial_collector.disable()
                trial_collector = candidate
                trial_keys.append(k)
        if len(trial_keys) > len(best_keys):
            if best_collector is not None:
                best_collector.disable()
            best_collector = trial_collector
            best_keys = trial_keys
            best_set = s
        elif trial_collector is not None:
            trial_collector.disable()
    collector = best_collector
    chosen = best_keys
    chosen_set = best_set
if collector is None:
    raise RuntimeError('No baseline PM Sampling metrics: '+repr(errors[-12:]))
print('Baseline config:',chosen,'set=',chosen_set,flush=True)
for cache_key in ('l1','l2'):
    print('CACHE', cache_key.upper(),
          'CONFIGURED' if cache_key in chosen else 'UNAVAILABLE (N/A)',flush=True)

# Extend only with metrics truly accepted together by the GPU PM sampler.
for k in ['fadd','fmul','ffma','tensor_active','tensor_inst','fma_inst']:
    c=configure(chosen+[k],chosen_set)
    if c:
        collector.disable()
        collector=c; chosen.append(k)
        print('Added:',k,flush=True)
    else:
        print('Not single-pass compatible:',k,flush=True)

# Six controlled load phases (seconds): 0-8 idle, 8-20 tensor ramp,
# 20-32 FP32 ramp, 32-42 bandwidth, 42-54 mixed, 54-60 cooldown.
# All reported FLOPS are actually executed GEMM operations (no synthetic data).
N=2048
GPU_FLOPS_PER_GEMM=2*N**3
# Disable TF32 to distinguish FP32 CUDA arithmetic from Tensor TF32.
torch.backends.cuda.matmul.allow_tf32=False
A=torch.randn((N,N),device='cuda',dtype=torch.float32)
B=torch.randn_like(A); O=torch.empty_like(A)
T=A.half().contiguous(); U=B.half().contiguous(); V=torch.empty_like(T)
# Streaming working set: three 16M float32 buffers (~192MiB total).
MEM_ELEMENTS=16*1024*1024
MA=torch.randn(MEM_ELEMENTS,device='cuda',dtype=torch.float32)
MB=torch.randn_like(MA); MO=torch.empty_like(MA)
for _ in range(4):
    torch.mm(A,B,out=O)
    torch.mm(T,U,out=V)
    torch.add(MA,MB,out=MO)
torch.cuda.synchronize()

PHASES=[
 {'start':0, 'end':8, 'name':'Idle', 'short':'IDLE', 'color':'#64748b'},
 {'start':8, 'end':20, 'name':'Tensor FP16 ramp', 'short':'FP16', 'color':'#a78bfa'},
 {'start':20, 'end':32, 'name':'CUDA FP32 ramp', 'short':'FP32', 'color':'#34d399'},
 {'start':32, 'end':42, 'name':'Memory bandwidth', 'short':'MEM', 'color':'#38bdf8'},
 {'start':42, 'end':54, 'name':'Mixed FP16 + FP32 + Memory', 'short':'MIX', 'color':'#f59e0b'},
 {'start':54, 'end':60, 'name':'Cooldown / Idle', 'short':'IDLE', 'color':'#64748b'},
]

# Per-second workload profile: nonuniform *actual GPU operations*.
# Duty budgets specify max host scheduling time, not assumed performance.
def load_profile(second):
    if second < 8 or second >= 54:
        return 'idle', {'fp32':0., 'tensor_fp16':0., 'memory':0.}
    if second < 20:
        step=min(int((second-8)//4),2)
        return 'tensor', {'fp32':0., 'tensor_fp16':[.12,.32,.72][step], 'memory':0.}
    if second < 32:
        step=min(int((second-20)//4),2)
        return 'fp32', {'fp32':[.12,.32,.72][step], 'tensor_fp16':0., 'memory':0.}
    if second < 42:
        step=min(int((second-32)//3),2)
        return 'memory', {'fp32':0., 'tensor_fp16':0., 'memory':[.15,.40,.78][step]}
    step=min(int((second-42)//4),2)
    return 'mixed', {'fp32':[.10,.18,.26][step],
                     'tensor_fp16':[.10,.18,.26][step],
                     'memory':[.10,.18,.26][step]}

# CPU monotonic timer controls slot duration; CUDA events measure executed GPU time.
def execute_budget(kind, fraction, slot_end):
    if fraction <= 0:
        return 0, 0., 0.
    budget=min(fraction*REPORT_INTERVAL_SECONDS,
               max(0.,slot_end-time.monotonic()))
    if budget <= .005:
        return 0,0.,0.
    deadline=min(slot_end,time.monotonic()+budget)
    count=0
    start_ev=torch.cuda.Event(enable_timing=True)
    stop_ev=torch.cuda.Event(enable_timing=True)
    start_ev.record()
    while time.monotonic() < deadline and count < 10000:
        if kind=='fp32':
            torch.mm(A,B,out=O)
        elif kind=='tensor_fp16':
            torch.mm(T,U,out=V)
        else:
            torch.add(MA,MB,out=MO)
        count += 1
        # limit queuing: accurate per-phase execution with no runaway backlog
        if count % 8 == 0:
            torch.cuda.synchronize()
    stop_ev.record();stop_ev.synchronize()
    elapsed=start_ev.elapsed_time(stop_ev)/1000
    return count, elapsed, time.monotonic()

started=False
throughput_points=[]
try:
    collector.start(); started=True
    capture_t0=time.monotonic()
    # One synchronized 1-second slot with a controlled intensity profile.
    # 'window_tflops' = actual executed GEMM FLOPs / slot wall time.
    # 'effective_tflops' = actual executed GEMM FLOPs / CUDA event time.
    # The first varies with scheduling duty cycle, the second with kernel efficiency.
    slot=0
    batches=0
    phase_points=[]
    while slot*REPORT_INTERVAL_SECONDS < CAPTURE_SECONDS:
        slot_start=capture_t0+slot*REPORT_INTERVAL_SECONDS
        slot_end=min(slot_start+REPORT_INTERVAL_SECONDS,
                     capture_t0+CAPTURE_SECONDS)
        now=time.monotonic()
        if now < slot_start:
            time.sleep(slot_start-now)
        if time.monotonic() >= slot_end:
            slot += 1
            continue
        phase,budgets=load_profile(slot*REPORT_INTERVAL_SECONDS)
        measurements={}
        # Alternation removes fixed ordering during mixed phases.
        kinds=['tensor_fp16','fp32','memory'] if slot%2==0 else ['fp32','tensor_fp16','memory']
        for kind in kinds:
            count, gpu_s, _=execute_budget(kind,budgets[kind],slot_end)
            measurements[kind]={'iterations':count,'kernel_seconds':gpu_s}
        duration=slot_end-slot_start
        for kind in ('tensor_fp16','fp32'):
            m=measurements[kind]
            flops=GPU_FLOPS_PER_GEMM*m['iterations']
            throughput_points.append({
                'time_s':slot*REPORT_INTERVAL_SECONDS+duration/2,
                'window':slot,'phase':phase,'type':kind,
                'iterations':m['iterations'],'kernel_seconds':m['kernel_seconds'],
                'window_seconds':duration,
                'window_tflops':flops/duration/1e12,
                'effective_tflops':flops/m['kernel_seconds']/1e12 if m['kernel_seconds']>0 else None,
            })
        phase_points.append({'window':slot,'phase':phase,
            'memory_iterations':measurements['memory']['iterations'],
            'memory_gpu_seconds':measurements['memory']['kernel_seconds']})
        batches += 1
        slot += 1
    collector.stop(); started=False
    # Summary per workload is the weighted-average effective FLOP/s.
    def summary(kind):
        pts=[x for x in throughput_points if x['type']==kind]
        total_seconds=sum(x['kernel_seconds'] for x in pts)
        total_flops=sum(2*N**3*x['iterations'] for x in pts)
        return {'seconds':total_seconds,'iterations':sum(x['iterations'] for x in pts),
                'effective_tflops':total_flops/total_seconds/1e12 if total_seconds>0 else None}
    fp32=summary('fp32')
    fp16=summary('tensor_fp16')
    data=collector.decode(max_samples=MAX_SAMPLES)
    print('Workload batches:',batches,'| requested capture:',CAPTURE_SECONDS,'seconds',flush=True)
    metrics=list(data.metrics)
    rows=[]
    for sample in data:
        st=int(sample.start_timestamp); en=int(sample.end_timestamp)
        if st and en>st:
            rows.append({'start_ns':st,'end_ns':en,
                         'values':[float(v) for v in sample.metric_values]})
finally:
    if started:
        try: collector.stop()
        except Exception: pass
    collector.disable()

out={'gpu':name,'chip':chip,'set':chosen_set,'chosen':chosen,'metrics':metrics,
     'samples':rows,'fp32_gemm':fp32,'fp16_gemm':fp16,'flops_points':throughput_points,
     'catalog_sizes':{k:len(v) for k,v in catalog.items()},'failed_attempts':len(errors),
     'phases':PHASES,'phase_points':phase_points}
with open(os.environ['GPU_PULSE_OUT'],'w') as f: json.dump(out,f)
print('Valid samples:',len(rows),'| metrics:',metrics,flush=True)
'''

env=os.environ.copy()
env['LD_LIBRARY_PATH']=str(pathlib.Path(lib_path).parent)+':'+env.get('LD_LIBRARY_PATH','')
env['GPU_PULSE_LIB']=lib_path
env['GPU_PULSE_OUT']=str(output_path)
env['GPU_PULSE_CAPTURE_SECONDS']=str(CAPTURE_SECONDS)
env['GPU_PULSE_SAMPLE_INTERVAL_NS']=str(SAMPLE_INTERVAL_NS)
env['GPU_PULSE_REPORT_INTERVAL_SECONDS']=str(REPORT_INTERVAL_SECONDS)
env['GPU_PULSE_MAX_SAMPLES']=str(MAX_SAMPLES)
r=subprocess.run([sys.executable,'-c',CHILD],env=env,text=True,capture_output=True)
print(r.stdout)
if r.returncode:
    print(r.stderr[-7000:])
    raise RuntimeError('실측 실패: 위 CUPTI 오류를 확인하세요.')
with open(output_path) as f: result=json.load(f)

metrics=result['metrics']; lookup={m:i for i,m in enumerate(metrics)}
keys={'gr':'gr__cycles_elapsed.max','sm':'sm__cycles_elapsed.sum',
      'read':'dramc__sectors_op_read.sum','write':'dramc__sectors_op_write.sum',
      'l1':'l1tex__t_sector_hit_rate.pct','l2':'lts__average_t_sector_hit_rate_realtime.pct',
      'fadd':'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum',
      'fmul':'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum',
      'ffma':'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum',
      'tensor_active':'sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed',
      'tensor_inst':'smsp__inst_executed_pipe_tensor.sum',
      'fma_inst':'smsp__inst_executed_pipe_fma.sum'}
rows=[]
for s in result['samples']:
    row={'time_ns':s['end_ns'],'dt_s':(s['end_ns']-s['start_ns'])/1e9}
    for k,m in keys.items():
        if m in lookup: row[k]=s['values'][lookup[m]]
    rows.append(row)
if not rows: raise RuntimeError('No valid CUPTI sample intervals')
df=pd.DataFrame(rows)
df['time_s']=(df.time_ns-df.time_ns.iloc[0])/1e9
df['window']=(df.time_s/REPORT_INTERVAL_SECONDS).astype(int)
for side in ('read','write'):
    if side in df: df[side+'_bytes']=df[side]*32

aggregations={'time_s':('time_s','mean'),'duration':('dt_s','sum')}
for k in keys:
    if k in df: aggregations[k]=(k,'sum' if k in ('fadd','fmul','ffma') else 'mean')
for side in ('read','write'):
    if side+'_bytes' in df: aggregations[side+'_bytes']=(side+'_bytes','sum')
agg=df.groupby('window').agg(**aggregations)
agg['time_s']=(agg.index.to_numpy()+0.5)*REPORT_INTERVAL_SECONDS
for side in ('read','write'):
    if side+'_bytes' in agg:
        agg[side+'_gbps']=agg[side+'_bytes']/agg.duration/1e9
if all(k in agg for k in ('fadd','fmul','ffma')):
    agg['hardware_fp32_tflops'] = (agg.fadd+agg.fmul+2*agg.ffma)/agg.duration/1e12
recent=agg.tail(10)
def recent_mean(k): return float(recent[k].mean()) if k in recent else None
def recent_rate(k):
    return float(recent[k+'_bytes'].sum()/recent.duration.sum()/1e9) if k+'_bytes' in recent else None
read_now=recent_rate('read');write_now=recent_rate('write')

# Hardware FP32 TFLOP/s requires all three SASS thread-level counters.
# This is an aggregate mixed-workload window, not the standalone FP32 test.
hardware_fp32=None
if all(k in df for k in ('fadd','fmul','ffma')):
    flops=(df.fadd+df.fmul+2*df.ffma).sum()
    duration=df.dt_s.sum()
    if duration>0: hardware_fp32=float(flops/duration/1e12)

def fmt(x,dec=2): return 'N/A' if x is None or not np.isfinite(x) else f'{x:,.{dec}f}'
DASHBOARD_WIDTH = 1080  # Same fixed outer width for cards and both Plotly charts

def card(label,value,unit,color='#60a5fa'):
    return f'<div style="background:#172033;border:1px solid #303c50;border-radius:12px;padding:16px;box-sizing:border-box;min-width:0">' \
           f'<div style="color:#9caac0;font-size:12px">{label}</div>' \
           f'<div style="color:{color};font-size:26px;font-weight:700;margin:8px 0">{value}</div>' \
           f'<div style="color:#8190a6;font-size:11px">{unit}</div></div>'

cards=[
 card('Tensor FP16 GEMM',fmt(result['fp16_gemm']['effective_tflops']),'active GEMM TFLOP/s · CUDA events','#a78bfa'),
 card('CUDA FP32 GEMM',fmt(result['fp32_gemm']['effective_tflops']),'active GEMM TFLOP/s · CUDA events','#a78bfa'),
 card('FP32 Hardware',fmt(hardware_fp32),'TFLOP/s · CUPTI counters','#60a5fa'),
 card('Tensor Pipe Active',fmt(recent_mean('tensor_active')),'% of peak · CUPTI','#fbbf24'),
 card('DRAM Read',fmt(read_now),'GB/s · CUPTI','#34d399'),
 card('DRAM Write',fmt(write_now),'GB/s · CUPTI','#34d399'),
 card('L1 Cache Hit',fmt(recent_mean('l1'),4),'raw .pct · CUPTI','#fbbf24'),
 card('L2 Cache Hit',fmt(recent_mean('l2')),'% · CUPTI','#fbbf24'),
 card('GR Cycles',fmt(recent_mean('gr'),0),'cycles/sample','#38bdf8'),
 card('SM Cycles',fmt(recent_mean('sm'),0),'cycles/sample','#38bdf8'),
 card('Tensor Instructions',fmt(recent_mean('tensor_inst'),0),'instructions/sample · not FLOPs','#f472b6'),
 card('FMA Instructions',fmt(recent_mean('fma_inst'),0),'instructions/sample · not FLOPs','#f472b6'),
]
display(HTML(f'<div style="background:#0b1220;color:white;padding:24px;border-radius:16px;font-family:Arial;box-sizing:border-box;width:{DASHBOARD_WIDTH}px;max-width:100%;margin:0 auto">'
    '<h2 style="margin:0">GPU PULSE</h2><p style="color:#91a0b7;font-size:12px">REAL HARDWARE COUNTERS + MEASURED GEMM THROUGHPUT</p>'
    '<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;width:100%">'+''.join(cards)+'</div>'
    f'<p style="color:#91a0b7;font-size:12px;margin:18px 0 0">GPU: {html.escape(result["gpu"])} | Chip: {html.escape(result["chip"])} | '
    f'Samples: {len(df)} | Duration: {df.dt_s.sum():.1f}s | Set: {html.escape(str(result["set"]))} | Real CUPTI / CUDA events</p></div>'))

# Display phase boundaries against the common elapsed-time axis.
phases=result['phases']
flops_df=pd.DataFrame(result.get('flops_points',[]))
if flops_df.empty:
    raise RuntimeError('No workload-window results. Check child process output.')

def add_phases(fig):
    for p in phases:
        fig.add_vrect(x0=p['start'],x1=p['end'],fillcolor=p['color'],
                      opacity=.105,line_width=0,layer='below')
        fig.add_annotation(x=(p['start']+p['end'])/2,y=1.045,yref='paper',
                           text=p['short'],showarrow=False,
                           font=dict(color=p['color'],size=11))
    fig.update_xaxes(range=[0,CAPTURE_SECONDS])

# FLOPS per 1-second time window explicitly includes scheduled idle time.
# Curves are real workload FLOPs / elapsed slot seconds, not simulated data.
fig_flops=go.Figure()
for kind,label,color in [
    ('tensor_fp16','Tensor FP16 | workload TFLOP/s','#a78bfa'),
    ('fp32','CUDA FP32 | workload TFLOP/s','#34d399')
]:
    subset=flops_df[flops_df['type']==kind].sort_values('time_s')
    fig_flops.add_trace(go.Scatter(
        x=subset.time_s,y=subset.window_tflops,mode='lines+markers',
        name=label,line=dict(color=color,width=2.5),marker=dict(size=5),
        customdata=subset[['iterations','kernel_seconds']].to_numpy(),
        hovertemplate='t=%{x:.1f}s<br>window TFLOP/s=%{y:.3f}'
                      '<br>GEMMs=%{customdata[0]}'
                      '<br>CUDA event=%{customdata[1]:.3f}s<extra>%{fullData.name}</extra>'
    ))
add_phases(fig_flops)
fig_flops.update_layout(
    title='GPU Pulse | Real GEMM FLOPs / 1-second window (varying workload)',
    xaxis_title='Capture time (s)',yaxis_title='Workload TFLOP/s · 1s wall window',
    template='plotly_dark',paper_bgcolor='#0b1220',plot_bgcolor='#111827',
    width=DASHBOARD_WIDTH,autosize=False,
    height=520,hovermode='x unified',
    legend=dict(orientation='h',x=.5,xanchor='center',y=-.20,yanchor='top',
                font=dict(size=11),traceorder='normal'),
    margin=dict(l=80,r=50,t=90,b=120))
fig_flops.show()

fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.10,
                  subplot_titles=['DRAM Read / Write throughput',
                                  'L1 / L2 cache hit rates',
                                  'Hardware FP32 FLOP/s (when counters available)'])
lines=[(1,'read_gbps','DRAM Read','#34d399'),(1,'write_gbps','DRAM Write','#60a5fa'),
       (2,'l1','L1 Hit','#fbbf24'),(2,'l2','L2 Hit','#a78bfa'),
       (3,'hardware_fp32_tflops','FP32 Hardware TFLOP/s','#60a5fa')]
for row,key,label,color in lines:
    if key in agg:
        fig.add_trace(go.Scatter(x=agg.time_s,y=agg[key],name=label,
                     mode='lines+markers',line=dict(color=color,width=2),
                     marker=dict(size=3)),row=row,col=1)
add_phases(fig)
fig.update_layout(title='GPU Pulse | CUPTI Hardware Timeline',
                  template='plotly_dark',paper_bgcolor='#0b1220',
                  plot_bgcolor='#111827',height=880,hovermode='x unified',
                  width=DASHBOARD_WIDTH,autosize=False,
                  legend=dict(orientation='h',x=.5,xanchor='center',y=-.13,yanchor='top',
                              font=dict(size=11),traceorder='normal'),
                  margin=dict(l=80,r=50,t=95,b=130))
fig.update_yaxes(title_text='GB/s',row=1,col=1)
fig.update_yaxes(title_text='Hit (%)',row=2,col=1)
fig.update_yaxes(title_text='TFLOP/s',row=3,col=1)
fig.update_xaxes(title_text='Capture time (s)',row=3,col=1)
fig.show()

print('Workload schedule:')
for p in phases:
    print(f"  {p['start']:>2}-{p['end']:>2}s  {p['name']}")
print('Zero FLOPS in idle/memory-only phases means no benchmark GEMMs were scheduled.')
print('FLOPS plot = actual executed GEMM FLOPs / 1-second wall window; NOT total GPU hardware FLOP/s.')
gpu_pulse_df=df;gpu_pulse_agg=agg;gpu_pulse_result=result;gpu_pulse_flops=flops_df
print('Unified interval:',REPORT_INTERVAL_SECONDS,'s | CUPTI:', SAMPLE_INTERVAL_NS,'ns')
print('Requested capture:', CAPTURE_SECONDS, 's | valid sample span:', round(float(df.time_s.max() - df.time_s.min()), 2), 's')
print('AVAILABLE COUNTERS:',result['chosen'])
print('FP32 hardware TFLOP/s:',fmt(hardware_fp32),'(only if 3 thread-level counters are available)')
print('Tensor effective FP16 GEMM:',fmt(result['fp16_gemm']['effective_tflops']),'TFLOP/s')
print('CUDA FP32 effective GEMM:',fmt(result['fp32_gemm']['effective_tflops']),'TFLOP/s')


# =============================================================
# GPU PULSE — PRECISION CAPABILITY & PERFORMANCE SWEEP
# Executes only genuinely available CUDA kernels.
# Unsupported formats are never silently converted to FP16/FP32.
# This is a separate POST-CAPTURE benchmark; it does not describe
# the 60-second CUPTI timeline above.
# =============================================================
import math as _gp_math
import traceback as _gp_tb
import shutil as _gp_shutil
import torch as _gp_torch

VECTOR_PRECISIONS = ['INT4','INT8','INT16','INT32','FP16','BF16','FP32','FP64']
MATRIX_PRECISIONS = ['INT1','INT2','INT4','INT8','MXFP4','NVFP4','MXFP6','FP8','MXFP8','FP16','BF16','FP32','TF32','FP64']
# Set additional typed runners here, if the correct vendor implementation
# is installed. They must RETURN a result with a measured CUDA event time.
# The runner signature is fn(device, event_timer) -> {"ops": int, "seconds": float, "backend": str}.
# Do not implement this with cast-to-FP16 emulation.
GPU_PULSE_NATIVE_RUNNERS = globals().get('GPU_PULSE_NATIVE_RUNNERS', {})

_gp_device = _gp_torch.device('cuda:0')
_gp_cap = _gp_torch.cuda.get_device_capability(0)
_gp_gpu = _gp_torch.cuda.get_device_name(0)
_gp_N = 512                 # fit FP64 and L4; adjust upward for peak sweeps
_gp_VEC_N = 2**22          # ~16 Mi elements; memory-bound rather than peak ALU
_gp_warmup = 3
_gp_reps = 12
_gp_results = []

def _gp_measure(work, work_ops, unit, backend, setup=None):
    """CUDA-event timed work; return best repeat, not a synthesized peak."""
    if setup is not None: setup()
    for _ in range(_gp_warmup): work()
    _gp_torch.cuda.synchronize()
    start = _gp_torch.cuda.Event(enable_timing=True)
    end = _gp_torch.cuda.Event(enable_timing=True)
    durations=[]
    for _ in range(_gp_reps):
        start.record();work();end.record();end.synchronize()
        durations.append(start.elapsed_time(end)/1000.0)
    sec=min(x for x in durations if x>0)
    return {'ops':int(work_ops),'seconds':float(sec),'backend':backend,'unit':unit,
            'rate':float(work_ops/sec/(1e12 if unit=='TFLOP/s' else 1e9))}

def _gp_add(family, precision, runner=None, reason='', native=True):
    entry={'family':family,'precision':precision,'status':'BACKEND_REQUIRED',
           'value':None,'unit':'','backend':'','detail':reason}
    if runner is not None:
        try:
            result=runner()
            if not isinstance(result,dict) or not result.get('seconds',0)>0:
                raise RuntimeError('Runner did not return measured execution time')
            if not result.get('backend'):
                raise RuntimeError('Runner did not identify a backend')
            unit=result.get('unit') or ('TFLOP/s' if precision.startswith(('FP','BF','TF','MX','NV')) else 'GOP/s')
            value=result.get('rate',result['ops']/result['seconds']/(1e12 if unit=='TFLOP/s' else 1e9))
            entry.update(status='MEASURED',value=float(value),unit=unit,
                         backend=result['backend'],detail='CUDA Events; executed kernel')
        except Exception as exc:
            entry.update(status='EXECUTION_FAILED',detail=str(exc)[:250])
    _gp_results.append(entry)

# Vector: a genuine GPU pointwise kernel, one addition per element.
# Bandwidth-limited throughputs in GOP/s or GFLOP/s; NOT tensor throughput.
_gp_vec_dtypes={
    'INT8':_gp_torch.int8,'INT16':_gp_torch.int16,'INT32':_gp_torch.int32,
    'FP16':_gp_torch.float16,'BF16':_gp_torch.bfloat16,
    'FP32':_gp_torch.float32,'FP64':_gp_torch.float64,
}
for _gp_p in VECTOR_PRECISIONS:
    _gp_key=('Vector',_gp_p)
    if _gp_key in GPU_PULSE_NATIVE_RUNNERS:
        try:
            _gp_add('Vector',_gp_p,runner=lambda k=_gp_key:GPU_PULSE_NATIVE_RUNNERS[k](_gp_device,_gp_measure))
        except Exception as e:
            _gp_add('Vector',_gp_p,reason=str(e))
    elif _gp_p=='INT4':
        _gp_add('Vector',_gp_p,reason='4-bit packed integer pointwise kernel required (no PyTorch int4 dtype).')
    else:
        def _vector_job(p=_gp_p):
            dt=_gp_vec_dtypes[p]
            n=_gp_VEC_N
            if dt.is_floating_point:
                a=_gp_torch.randn(n,device=_gp_device,dtype=dt)
                b=_gp_torch.randn(n,device=_gp_device,dtype=dt)
            else:
                a=_gp_torch.randint(0,16,(n,),device=_gp_device,dtype=dt)
                b=_gp_torch.randint(0,16,(n,),device=_gp_device,dtype=dt)
            out=_gp_torch.empty_like(a)
            return _gp_measure(lambda:_gp_torch.add(a,b,out=out),n,
                               'GFLOP/s' if dt.is_floating_point else 'GOP/s',
                               'PyTorch ATen CUDA pointwise add')
        _gp_add('Vector',_gp_p,runner=_vector_job)

# Matrix: actual CUDA GEMM; 2MNK counts multiply-add as two ops.
# Do not label INT8 tensor MM as 'CUDA int8 vector'.
_gp_mat_dtypes={'FP16':_gp_torch.float16,'BF16':_gp_torch.bfloat16,
                'FP32':_gp_torch.float32,'FP64':_gp_torch.float64}
for _gp_p in MATRIX_PRECISIONS:
    _gp_key=('Matrix',_gp_p)
    if _gp_key in GPU_PULSE_NATIVE_RUNNERS:
        _gp_add('Matrix',_gp_p,runner=lambda k=_gp_key:GPU_PULSE_NATIVE_RUNNERS[k](_gp_device,_gp_measure))
    elif _gp_p in _gp_mat_dtypes or _gp_p=='TF32':
        def _matrix_job(p=_gp_p):
            dt=_gp_mat_dtypes.get(p,_gp_torch.float32)
            n=_gp_N
            a=_gp_torch.randn((n,n),device=_gp_device,dtype=dt)
            b=_gp_torch.randn_like(a); out=_gp_torch.empty_like(a)
            old_tf32=_gp_torch.backends.cuda.matmul.allow_tf32
            try:
                # TF32 is a Tensor Core compute mode on FP32 storage,
                # not a distinct torch.dtype. Backend selection should
                # be verified using Nsight Compute to claim tensor instructions.
                _gp_torch.backends.cuda.matmul.allow_tf32=(p=='TF32')
                return _gp_measure(lambda:_gp_torch.mm(a,b,out=out),2*n**3,
                                   'TFLOP/s',
                                   'cuBLAS via PyTorch (TF32 policy)' if p=='TF32' else 'cuBLAS via PyTorch')
            finally:
                _gp_torch.backends.cuda.matmul.allow_tf32=old_tf32
        _gp_add('Matrix',_gp_p,runner=_matrix_job)
    elif _gp_p=='INT8':
        def _int8_job():
            if not hasattr(_gp_torch,'_int_mm'):
                raise RuntimeError('torch._int_mm not exposed by this PyTorch build')
            n=_gp_N
            a=_gp_torch.randint(-4,4,(n,n),device=_gp_device,dtype=_gp_torch.int8)
            b=_gp_torch.randint(-4,4,(n,n),device=_gp_device,dtype=_gp_torch.int8)
            return _gp_measure(lambda:_gp_torch._int_mm(a,b),2*n**3,'GOP/s',
                               'PyTorch CUDA _int_mm (int32 accumulation)')
        _gp_add('Matrix',_gp_p,runner=_int8_job)
    elif _gp_p=='FP8':
        def _fp8_job():
            if _gp_cap[0]<9: raise RuntimeError('Native FP8 tensor GEMM generally requires Hopper or later (SM90+)')
            if not hasattr(_gp_torch,'_scaled_mm') or not hasattr(_gp_torch,'float8_e4m3fn'):
                raise RuntimeError('PyTorch _scaled_mm / float8_e4m3fn API unavailable')
            n=_gp_N
            x=(_gp_torch.randn((n,n),device=_gp_device)*0.1).to(_gp_torch.float8_e4m3fn)
            y=(_gp_torch.randn((n,n),device=_gp_device)*0.1).to(_gp_torch.float8_e4m3fn)
            s=_gp_torch.tensor(1.0,device=_gp_device,dtype=_gp_torch.float32)
            def work():
                try: return _gp_torch._scaled_mm(x,y,scale_a=s,scale_b=s,out=None)
                except TypeError: return _gp_torch._scaled_mm(x,y,s,s)
            return _gp_measure(work,2*n**3,'TFLOP/s','PyTorch CUDA _scaled_mm E4M3')
        _gp_add('Matrix',_gp_p,runner=_fp8_job)
    else:
        reasons={
            'INT1':'Packed binary matrix MMA requires dedicated CUTLASS/CUDA kernel; not equivalent to torch.int8 GEMM.',
            'INT2':'Packed 2-bit matrix MMA requires dedicated CUTLASS/CUDA kernel.',
            'INT4':'Packed 4-bit matrix MMA requires dedicated CUTLASS/CUDA kernel.',
            'MXFP4':'Scaled FP4 Tensor Core requires compatible Blackwell+ CUTLASS/cuBLASLt kernel.',
            'NVFP4':'NVFP4 Tensor Core requires compatible Blackwell+ backend with correct per-block scaling.',
            'MXFP6':'Block-scaled FP6 requires a matching GPU/backend kernel.',
            'MXFP8':'Block-scaled FP8 requires dedicated scaled MMA, distinct from plain FP8.',
        }
        _gp_add('Matrix',_gp_p,reason=reasons.get(_gp_p,'Native backend not implemented'))

precision_results=pd.DataFrame(_gp_results)
print('\n=== GPU PULSE | GPU-VERIFIED PRECISION EXECUTION ===')
print('GPU:',_gp_gpu,'SM:',_gp_cap,'| backend-specific native execution only')
print(precision_results[['family','precision','status','value','unit','backend','detail']].to_string(index=False,max_colwidth=70))

# Native-support coverage is based on successful execution, not architecture guesswork.
from plotly.subplots import make_subplots as _gp_subplots
fig_precision=_gp_subplots(rows=2,cols=2,
    subplot_titles=['Vector floating-point · GFLOP/s','Vector integer · GOP/s',
                    'Matrix floating-point · TFLOP/s','Matrix integer · GOP/s'],
    vertical_spacing=.18,horizontal_spacing=.13)
for (family,unit,row,col,color) in [
    ('Vector','GFLOP/s',1,1,'#60a5fa'),
    ('Vector','GOP/s',1,2,'#34d399'),
    ('Matrix','TFLOP/s',2,1,'#a78bfa'),
    ('Matrix','GOP/s',2,2,'#fbbf24')]:
    part=precision_results[(precision_results.family==family)&
                            (precision_results.unit==unit)&
                            (precision_results.status=='MEASURED')]
    fig_precision.add_trace(go.Bar(x=part.precision,y=part.value,
         name=f'{family} {unit}',marker_color=color,
         text=[f'{v:.2f}' for v in part.value],textposition='outside',
         hovertemplate='%{x}: %{y:.3f} '+unit+'<extra></extra>'),row=row,col=col)
fig_precision.update_layout(title='GPU Pulse | Actually Executed Precision Kernels',
    width=DASHBOARD_WIDTH,autosize=False,height=780,showlegend=False,
    template='plotly_dark',paper_bgcolor='#0b1220',plot_bgcolor='#111827',
    margin=dict(l=80,r=50,t=95,b=90))
fig_precision.update_yaxes(title_text='GFLOP/s',row=1,col=1)
fig_precision.update_yaxes(title_text='GOP/s',row=1,col=2)
fig_precision.update_yaxes(title_text='TFLOP/s',row=2,col=1)
fig_precision.update_yaxes(title_text='GOP/s',row=2,col=2)
fig_precision.show()

# =============================================================
# 60s CUPTI L1 / L2 CACHE RECAP
# Kept visible AFTER the precision sweep. No additional GPU
# operations are run; this reuses the real CUPTI capture above.
# =============================================================
_cache_l1_available = 'l1' in agg.columns and agg['l1'].notna().any()
_cache_l2_available = 'l2' in agg.columns and agg['l2'].notna().any()

def _cache_last_valid(col):
    if col not in agg.columns:
        return None
    vals = pd.to_numeric(agg[col], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.iloc[-1]) if len(vals) else None

_cache_l1 = _cache_last_valid('l1')
_cache_l2 = _cache_last_valid('l2')
_cache_msg = (
    'CUPTI PM Sampling · 1-second reporting windows · captured during 60-second workload. '
    'Unavailable counters are N/A, not 0. Percentages are unweighted means '
    'of sampled metric values, not sector-weighted aggregate hit rates.'
)
display(HTML(
    f'<div style="background:#0b1220;color:#edf2fc;border-radius:14px;'
    f'padding:20px;width:{DASHBOARD_WIDTH}px;max-width:100%;box-sizing:border-box;'
    f'margin:12px auto;font-family:Arial,sans-serif">'
    f'<div style="font-size:19px;font-weight:700;margin-bottom:14px">GPU PULSE | L1 / L2 CACHE</div>'
    f'<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px">'
    f'{card("L1 / TEX Cache Hit",fmt(_cache_l1,4),"% · CUPTI .pct (unweighted mean)","#fbbf24")}'
    f'{card("L2 Cache Hit",fmt(_cache_l2,2),"% · CUPTI .pct (unweighted mean)","#a78bfa")}'
    f'</div><p style="color:#91a0b7;font-size:12px;margin:15px 0 0">{_cache_msg}</p>'
    f'</div>'
))
fig_cache = go.Figure()
for col,label,color in [
    ('l1','L1 / TEX Cache Hit','#fbbf24'),
    ('l2','L2 Cache Hit','#a78bfa'),
]:
    if col in agg.columns and agg[col].notna().any():
        fig_cache.add_trace(go.Scatter(
            x=agg.time_s,y=agg[col],mode='lines+markers',name=label,
            line=dict(color=color,width=2.5),marker=dict(size=5),
            hovertemplate='t=%{x:.1f}s<br>Hit rate=%{y:.4f}%<extra>%{fullData.name}</extra>'
        ))
    else:
        print(f'{label}: N/A — not available in this GPU/CUPTI single-pass metric set')
add_phases(fig_cache)
if not fig_cache.data:
    fig_cache.add_annotation(x=.5,y=.5,xref='paper',yref='paper',
        text='L1/L2 cache metrics unavailable in this capture',showarrow=False,
        font=dict(size=15,color='#b6c4da'))
fig_cache.update_layout(
    title='GPU Pulse | L1 / L2 Cache Hit · 1-second CUPTI Timeline',
    template='plotly_dark',paper_bgcolor='#0b1220',plot_bgcolor='#111827',
    width=DASHBOARD_WIDTH,autosize=False,height=475,
    xaxis_title='Capture time (seconds)',yaxis_title='Cache hit (%)',
    hovermode='x unified',
    legend=dict(orientation='h',x=.5,xanchor='center',y=-.20,yanchor='top',font=dict(size=11)),
    margin=dict(l=80,r=50,t=90,b=115)
)
fig_cache.show()
gpu_pulse_cache_agg = agg[['time_s'] + [c for c in ('l1','l2') if c in agg.columns]].copy()

print('\nMissing formats are visible in precision_results with explicit reasons.')
print('The precision benchmark runs AFTER the 60-second capture; it does not change its CUPTI timeline.')
print('Matrix TF32 is only a math-policy test unless its actual MMA instruction is independently verified.')
print('Tensor formats with scaling are NOT emulated with FP16.')
gpu_pulse_precision_results=precision_results