"use client";

import { Leva } from "leva";
import { Canvas } from "@react-three/fiber";
import {
  AccumulativeShadows,
  RandomizedLight,
  OrbitControls,
} from "@react-three/drei";
import { Environment } from "@/components/environment";
import { HumanMesh } from "@/components/human-mesh";
import { SquatAnalysis } from "@/components/squat-analysis";
import { useState } from "react";
import { BenchAnalysis } from "@/components/bench-analysis";
import { DeadliftAnalysis } from "@/components/deadlift-analysis";

export default function App() {
  const [type, setType] = useState<"squat" | "bench" | "deadlift">("bench");
  const [frameIndex, setFrameIndex] = useState(0);
  const IMG_0026 = Array.from(
    { length: 530 },
    (_, i) =>
      `/IMG_0026/mesh_predicted_frame_${String(i).padStart(3, "0")}_mesh_000.ply`,
  );
  const IMG_0027 = Array.from(
    { length: 456 },
    (_, i) =>
      `/IMG_0027/mesh_predicted_frame_${String(i).padStart(3, "0")}_mesh_000.ply`,
  );
  const IMG_0028 = Array.from(
    { length: 207 },
    (_, i) =>
      `/IMG_0028/mesh_predicted_frame_${String(i).padStart(3, "0")}_mesh_000.ply`,
  );

  const currentVideo =
    type === "squat" ? IMG_0026 : type === "bench" ? IMG_0028 : IMG_0027;

  return (
    <div style={{ width: "100%", height: "100vh" }}>
      <Canvas camera={{ position: [0, 0, 5], fov: 50 }}>
        <group position={[0, -0.65, 0]}>
          <HumanMesh urls={currentVideo} onFrameChange={setFrameIndex} />
          <AccumulativeShadows
            temporal
            frames={200}
            color="black"
            colorBlend={0.5}
            opacity={1}
            scale={50}
            alphaTest={0.85}
          >
            <RandomizedLight
              amount={8}
              radius={5}
              ambient={0.5}
              position={[5, 3, 2]}
              bias={0.001}
            />
          </AccumulativeShadows>
        </group>
        <Environment />
        <OrbitControls
          enablePan={false}
          enableZoom={false}
          minPolarAngle={Math.PI / 2.1}
          maxPolarAngle={Math.PI / 2.1}
        />
      </Canvas>
      {type === "squat" && (
        <SquatAnalysis
          frameIndex={frameIndex}
          videoName="IMG_0026"
          totalFrames={530}
        />
      )}
      {type === "bench" && (
        <BenchAnalysis
          frameIndex={frameIndex}
          videoName="IMG_0028"
          totalFrames={207}
        />
      )}
      {type === "deadlift" && (
        <DeadliftAnalysis
          frameIndex={frameIndex}
          videoName="IMG_0027"
          totalFrames={456}
        />
      )}

      <Leva />
    </div>
  );
}
