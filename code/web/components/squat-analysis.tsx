"use client";

import { useEffect, useMemo, useState } from "react";

const JOINTS = {
  LEFT_HIP: 1,
  RIGHT_HIP: 2,
  LEFT_KNEE: 4,
  RIGHT_KNEE: 5,
  LEFT_SHOULDER: 7,
  RIGHT_SHOULDER: 8,
};

interface Keypoint3D {
  x: number;
  y: number;
  z: number;
}

interface SquatMetrics {
  depthOk: boolean;
  hipAngle: number;
  kneeAngle: number;
  trunkLean: number;
  symmetry: number;
}

function parseKeypoints(data: any): Keypoint3D[] | null {
  if (!data || !data[0]?.pred_keypoints_3d) return null;
  const raw = data[0].pred_keypoints_3d;
  return raw.map(([x, y, z]: number[]) => ({ x, y, z }));
}

function angleBetween(a: Keypoint3D, b: Keypoint3D, c: Keypoint3D): number {
  const ab = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const cb = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  const dot = ab.x * cb.x + ab.y * cb.y + ab.z * cb.z;
  const magAB = Math.sqrt(ab.x ** 2 + ab.y ** 2 + ab.z ** 2);
  const magCB = Math.sqrt(cb.x ** 2 + cb.y ** 2 + cb.z ** 2);
  return (Math.acos(dot / (magAB * magCB)) * 180) / Math.PI;
}

function computeMetrics(kp: Keypoint3D[]): SquatMetrics {
  const lHip = kp[JOINTS.LEFT_HIP];
  const rHip = kp[JOINTS.RIGHT_HIP];
  const lKnee = kp[JOINTS.LEFT_KNEE];
  const rKnee = kp[JOINTS.RIGHT_KNEE];
  const lSho = kp[JOINTS.LEFT_SHOULDER];
  const rSho = kp[JOINTS.RIGHT_SHOULDER];

  // Depth
  const hipY = (lHip.y + rHip.y) / 2;
  const kneeY = (lKnee.y + rKnee.y) / 2;
  const depthOk = hipY < kneeY;

  // Angle of the left knee
  const kneeAngle = angleBetween(lHip, lKnee, rKnee);

  // Upper body tilt
  const midHip = {
    x: (lHip.x + rHip.x) / 2,
    y: (lHip.y + rHip.y) / 2,
    z: (lHip.z + rHip.z) / 2,
  };
  const midSho = {
    x: (lSho.x + rSho.x) / 2,
    y: (lSho.y + rSho.y) / 2,
    z: (lSho.z + rSho.z) / 2,
  };
  const vertical = { x: midHip.x, y: midHip.y + 1, z: midHip.z };
  const trunkLean = angleBetween(vertical, midHip, midSho);

  // Symmetry
  const symmetry = Math.abs(lHip.y - rHip.y) * 100;

  // Hip angle
  const hipAngle = angleBetween(midSho, midHip, {
    x: midHip.x,
    y: midHip.y - 1,
    z: midHip.z,
  });

  return { depthOk, hipAngle, kneeAngle, trunkLean, symmetry };
}

interface SquatAnalysisProps {
  frameIndex: number;
  videoName: string;
  totalFrames: number;
}

export function SquatAnalysis({
  frameIndex,
  videoName,
  totalFrames,
}: SquatAnalysisProps) {
  const [allMetrics, setAllMetrics] = useState<(SquatMetrics | null)[]>([]);
  const [loading, setLoading] = useState(true);

  // Load all JSON files once
  useEffect(() => {
    const fetchAll = async () => {
      const results = await Promise.all(
        Array.from({ length: totalFrames }, async (_, i) => {
          const idx = String(i).padStart(3, "0");
          try {
            const r = await fetch(
              `/${videoName}/mesh_predicted_frame_${idx}_keypoints.json`,
            );
            const data = await r.json();
            const kp = parseKeypoints(data);
            return kp ? computeMetrics(kp) : null;
          } catch {
            return null;
          }
        }),
      );
      setAllMetrics(results);
      setLoading(false);
    };
    fetchAll();
  }, [videoName, totalFrames]);

  // Overall metrics
  const globalValidation = useMemo(() => {
    const valid = allMetrics.filter(Boolean) as SquatMetrics[];
    if (!valid.length) return null;

    // Depth validation
    const depthReached = valid.some((m) => m.depthOk);

    // Maximum upper body tilt
    const maxTrunkLean = Math.max(...valid.map((m) => m.trunkLean));

    // Maximum symmetry
    const maxSymmetry = Math.max(...valid.map((m) => m.symmetry));

    return { depthReached, maxTrunkLean, maxSymmetry };
  }, [allMetrics]);

  const currentMetrics = allMetrics[frameIndex];

  if (loading)
    return <div style={overlayStyle}>Chargement des métriques...</div>;

  return (
    <div style={overlayStyle}>
      <div style={{ fontWeight: "bold", marginBottom: 10, fontSize: 15 }}>
        Analyse squat
      </div>
      {currentMetrics && (
        <>
          <Row
            label="Profondeur"
            value={currentMetrics.depthOk ? "✅ Sous le genou" : "⬆️ Au-dessus"}
            ok={currentMetrics.depthOk}
          />
          <Row
            label="Inclinaison buste"
            value={`${currentMetrics.trunkLean.toFixed(1)}°`}
            ok={currentMetrics.trunkLean < 45}
          />
          <Row
            label="Symétrie"
            value={`${currentMetrics.symmetry.toFixed(1)} mm`}
            ok={currentMetrics.symmetry < 20}
          />
        </>
      )}

      {globalValidation && (
        <>
          <Row
            label="Profondeur IPF"
            value={
              globalValidation.depthReached ? "✅ Valide" : "❌ Insuffisante"
            }
            ok={globalValidation.depthReached}
          />
          <Row
            label="Inclinaison max"
            value={`${globalValidation.maxTrunkLean.toFixed(1)}°`}
            ok={globalValidation.maxTrunkLean < 45}
          />
          <Row
            label="Asymétrie max"
            value={`${globalValidation.maxSymmetry.toFixed(1)} mm`}
            ok={globalValidation.maxSymmetry < 20}
          />
        </>
      )}
    </div>
  );
}

const overlayStyle: React.CSSProperties = {
  position: "absolute",
  top: 16,
  right: 16,
  background: "rgba(0,0,0,0.75)",
  color: "white",
  borderRadius: 12,
  padding: "16px 20px",
  fontFamily: "monospace",
  fontSize: 13,
  minWidth: 220,
  backdropFilter: "blur(8px)",
};

function Row({
  label,
  value,
  ok,
}: {
  label: string;
  value: string;
  ok: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        marginBottom: 6,
        gap: 16,
      }}
    >
      <span style={{ color: "#aaa" }}>{label}</span>
      <span style={{ color: ok ? "#4ade80" : "#f87171", fontWeight: "bold" }}>
        {value}
      </span>
    </div>
  );
}
