import {useEffect,useState} from 'react';
type Profile={subject_id:string;data_type:string;heart_rate_bpm:number;systolic_bp_mmHg:number;diastolic_bp_mmHg:number;spo2_pct:number;skin_temperature_c:number;bone_density_g_cm2:number;bone_baseline_g_cm2:number;sleep_hours:number;note:string};
export default function HealthDashboard(){
 const [p,setP]=useState<Profile|null>(null);
 useEffect(()=>{fetch('/api/demo-profile').then(x=>x.json()).then(setP).catch(()=>setP(null))},[]);
 return <div className="panel"><div className="section-heading">Astronaut physiology <span className="flag">SYNTHETIC DEMO</span></div>
  {p?<><div className="health-grid">
    <span>Heart rate <b>{p.heart_rate_bpm} bpm</b></span><span>Blood pressure <b>{p.systolic_bp_mmHg}/{p.diastolic_bp_mmHg} mmHg</b></span>
    <span>Blood oxygen <b>{p.spo2_pct}%</b></span><span>Skin temperature <b>{p.skin_temperature_c} °C</b></span>
    <span>Bone density <b>{p.bone_density_g_cm2} g/cm²</b></span><span>Earth BMD baseline <b>{p.bone_baseline_g_cm2} g/cm²</b></span>
    <span>Sleep <b>{p.sleep_hours} hours</b></span><span>Gravity prescription <b>NOT VALIDATED</b></span>
  </div><p className="hint">{p.note}</p></>:<p className="hint">Demo profile unavailable; no real health observations loaded.</p>}
 </div>;
}
