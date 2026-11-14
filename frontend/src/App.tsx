import {useEffect,useState,useRef} from 'react';
import Spacecraft3D,{type Frame} from './components/Spacecraft3D';
import HealthDashboard from './components/HealthDashboard';
type Result={samples:Frame[];summary:{mean_abs_error_g:number;max_abs_error_g:number;torque_saturation_samples:number};source:string};
export default function App(){
 const [radius,setRadius]=useState(100),[target,setTarget]=useState(1),[peak,setPeak]=useState(.8),[duration,setDuration]=useState(1200);
 const [data,setData]=useState<Result|null>(null),[idx,setIdx]=useState(0),[playing,setPlaying]=useState(false),[error,setError]=useState('');
 const last=useRef<number|null>(null);
 useEffect(()=>{const abort=new AbortController();setError('');setPlaying(false);
 fetch('/api/simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({radius_m:radius,target_gravity_g:target,peak_thrust_g:peak,duration_s:duration}),signal:abort.signal})
 .then(async r=>{if(!r.ok)throw new Error(await r.text());return r.json()}).then((v:Result)=>{setData(v);setIdx(0)}).catch(e=>{if(e.name!=='AbortError')setError(String(e))});return()=>abort.abort();
 },[radius,target,peak,duration]);
 useEffect(()=>{if(!playing||!data)return;let req=0;
 function tick(t:number){if(last.current!==null){const delta=Math.min(90,t-last.current);setIdx(i=>Math.min((data?.samples.length??1)-1,i+Math.max(1,Math.round(delta/35))))};last.current=t;req=requestAnimationFrame(tick)}
 req=requestAnimationFrame(tick);return()=>{last.current=null;cancelAnimationFrame(req)}},[playing,data]);
 const s=data?.samples[idx]??null;
 return <main className="page"><div className="header"><div><div className="eyebrow">COMPUTATIONAL RESEARCH • SIMULATION ONLY</div><h1>Personalized Artificial Gravity</h1><p>Closed-loop ring dynamics, astronaut telemetry, and experimental physiology</p></div><div className="logo">G<span>·</span>AI</div></div>
 <div className="layout"><section className="main-panel"><div className="panel"><div className="section-heading">Rotating spacecraft <span className="flag">3D • PHYSICS BASED</span></div><Spacecraft3D frame={s}/><div className="hint">Drag to orbit · Scroll to zoom · Blue ring spins about X axis · Yellow marker is the astronaut.</div></div>
 <div className="metrics">{[
  ['Actual gravity',s?.crew_gravity_g?.toFixed(3)+' G'],['Target gravity',target.toFixed(2)+' G'],['Ring RPM',s?.ring_rpm?.toFixed(3)],['Target RPM',s?.target_rpm?.toFixed(3)??'N/A'],
  ['Motor torque',((s?.applied_motor_torque_nm??0)/1e6).toFixed(2)+' MN·m'],['Engine acceleration',((s?.axial_accel_m_s2??0)/9.80665).toFixed(2)+' G']
 ].map(([name,value])=><div className="metric" key={name}><small>{name}</small><strong>{value==='undefined G'?'—':value}</strong></div>)}</div>
 <div className="panel"><div className="section-heading">Mission timeline <span className="hint">{s?.time_s?.toFixed(0)??'0'} s / {duration}s</span></div>
 <input type="range" min={0} max={Math.max(0,(data?.samples.length??1)-1)} value={idx} onChange={e=>setIdx(+e.target.value)}/>
 <div className="actions"><button onClick={()=>setPlaying(v=>!v)}>{playing?'Ⅱ Pause':'▶ Play'}</button><button onClick={()=>{setIdx(0);setPlaying(false)}}>↺ Reset</button><span className="hint">First 30% accelerate → 40% coast → 30% brake</span></div>
 </div></section>
 <aside className="side-panel"><div className="panel"><div className="section-heading">Mission parameters</div>
 {([["Ring radius (m)",radius,setRadius,20,500,10],["Gravity target (G)",target,setTarget,.3,1.5,.05],["Peak thrust (G)",peak,setPeak,0,1,.05],["Duration (s)",duration,setDuration,240,3600,120]] as const).map(([title,v,fn,min,max,step])=><label key={title}>{title} <b>{v}</b><input type="range" min={min} max={max} step={step} value={v} onChange={e=>fn(+e.target.value)}/></label>)}
 <p className="hint">Steady-state reference with torque-limited motor tracking. Recorded motion is simulated, not measured.</p>
 </div><HealthDashboard/><div className="panel"><div className="section-heading">Scientific status</div>
 <p className="hint">Physiological training requires actual participant-level datasets. Catalog files alone do not establish a gravity-to-health response.</p>
 <span className="flag">NO VALIDATED PERSONALIZED PRESCRIPTION</span>
 {data&&<p className="hint">Mean absolute gravity error: {data.summary.mean_abs_error_g.toFixed(4)} G</p>}
 {error&&<p className="error">Backend error: {error}</p>}</div></aside></div>
 </main>
}
