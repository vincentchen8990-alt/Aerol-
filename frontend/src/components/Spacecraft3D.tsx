import {Canvas} from '@react-three/fiber';
import {OrbitControls,Grid,Stars,Text} from '@react-three/drei';
import * as THREE from 'three';
import {useMemo} from 'react';

export type Frame = {time_s:number;render_ring_angle_rad:number;ring_rpm:number;hub_angle_rad:number;
 crew_gravity_g:number;crew_gravity_radial_m_s2:number;crew_gravity_tangential_m_s2:number;crew_gravity_axial_m_s2:number;
 axial_accel_m_s2:number;target_rpm:number|null;position_m:number;applied_motor_torque_nm:number;};
function Model({frame}:{frame:Frame|null}){
 const spokes=useMemo(()=>Array.from({length:12},(_,i)=>i*Math.PI/6),[]);
 return <group>
  {/* central hub/longitudinal spacecraft axis is the X axis */}
  <mesh rotation={[0,0,Math.PI/2]}><cylinderGeometry args={[.23,.23,6,24]}/><meshStandardMaterial color="#74859c" metalness={.6} roughness={.35}/></mesh>
  <mesh position={[-3.2,0,0]} rotation={[0,0,Math.PI/2]}><coneGeometry args={[.48,.9,32]}/><meshStandardMaterial color="#f59e0b" emissive="#78350f"/></mesh>
  <mesh position={[3.15,0,0]} rotation={[0,0,Math.PI/2]}><coneGeometry args={[.2,.65,24]}/><meshStandardMaterial color="#475569"/></mesh>
  <group rotation={[frame?.render_ring_angle_rad??0,0,0]}>
    <mesh rotation={[0,Math.PI/2,0]}><torusGeometry args={[2,.20,18,120]}/><meshStandardMaterial color="#38bdf8" metalness={.5} roughness={.26}/></mesh>
    <mesh rotation={[0,Math.PI/2,0]}><torusGeometry args={[1.69,.015,8,120]}/><meshStandardMaterial color="#c6ecff" transparent opacity={.7}/></mesh>
    {spokes.map((a,i)=>{
       const end=new THREE.Vector3(0,Math.cos(a),Math.sin(a));
       return <mesh key={i} position={[0,end.y,end.z]} rotation={[Math.atan2(-end.z,end.y),0,0]}>
           <cylinderGeometry args={[.025,.025,2.0,8]}/><meshStandardMaterial color="#73859b"/>
       </mesh>;
     })}
    <mesh position={[0,2,0]}><sphereGeometry args={[.15,24,24]}/><meshStandardMaterial color="#fde047" emissive="#854d0e"/></mesh>
    <Text position={[.35,2.26,0]} fontSize={.22} color="#facc15">Crew</Text>
    <arrowHelper args={[new THREE.Vector3(0,1,0),new THREE.Vector3(0,2,0), .95,0x22c55e]}/>
  </group>
  <mesh><sphereGeometry args={[.28,24,24]}/><meshStandardMaterial color="#64748b" metalness={.7}/></mesh>
 </group>
}
export default function Spacecraft3D({frame}:{frame:Frame|null}){
 return <div className="canvas-frame"><Canvas camera={{position:[5.5,3.9,6.5],fov:45}}>
  <color attach="background" args={['#070e1b']}/><ambientLight intensity={.65}/><directionalLight position={[5,8,10]} intensity={2}/>
  <Stars radius={100} depth={50} count={600} factor={3} saturation={0} fade speed={.5}/>
  <Grid infiniteGrid fadeDistance={20} cellColor="#17334a" sectionColor="#164157" position={[0,-3.3,0]} />
  <Model frame={frame}/><OrbitControls makeDefault minDistance={3} maxDistance={22}/>
 </Canvas></div>
}
