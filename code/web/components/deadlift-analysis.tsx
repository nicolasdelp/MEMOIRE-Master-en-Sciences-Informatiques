"use client";

import { useEffect, useMemo, useState } from "react";

const JOINTS = {
  LEFT_HIP: 1,
  RIGHT_HIP: 2,
  LEFT_KNEE: 4,
  RIGHT_KNEE: 5,
  LEFT_SHOULDER: 7,
  RIGHT_SHOULDER: 8,
  LEFT_ELBOW: 9,
  RIGHT_ELBOW: 10,
  LEFT_WRIST: 11,
  RIGHT_WRIST: 12,
};

interface Keypoint3D {
  x: number;
  y: number;
  z: number;
}

interface DeadliftMetrics {
  kneesLocked: boolean;
  shouldersBack: boolean;
  hipAngle: number;
  kneeAngle: number;
  trunkLean: number;
  symmetry: number;
  leftWrist: Keypoint3D;
  rightWrist: Keypoint3D;
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
  return (
    (Math.acos(Math.max(-1, Math.min(1, dot / (magAB * magCB)))) * 180) /
    Math.PI
  );
}

function dist3D(a: Keypoint3D, b: Keypoint3D): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
}

function computeMetrics(kp: Keypoint3D[]): DeadliftMetrics {
  const lHip = kp[JOINTS.LEFT_HIP];
  const rHip = kp[JOINTS.RIGHT_HIP];
  const lKnee = kp[JOINTS.LEFT_KNEE];
  const rKnee = kp[JOINTS.RIGHT_KNEE];
  const lSho = kp[JOINTS.LEFT_SHOULDER];
  const rSho = kp[JOINTS.RIGHT_SHOULDER];
  const lWrist = kp[JOINTS.LEFT_WRIST];
  const rWrist = kp[JOINTS.RIGHT_WRIST];

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
  const midKnee = {
    x: (lKnee.x + rKnee.x) / 2,
    y: (lKnee.y + rKnee.y) / 2,
    z: (lKnee.z + rKnee.z) / 2,
  };

  // Knee angle
  const floorKnee = { x: midKnee.x, y: midKnee.y - 1, z: midKnee.z };
  const kneeAngle = angleBetween(midHip, midKnee, floorKnee);

  // Knees locked
  const lKneeAngle = angleBetween(lHip, lKnee, rKnee);
  const kneesLocked = lKneeAngle > 160;

  // Trunk lean
  const vertical = { x: midHip.x, y: midHip.y + 1, z: midHip.z };
  const trunkLean = angleBetween(vertical, midHip, midSho);

  // Shoulders back
  const shouldersBack = midSho.z <= midHip.z;

  // Hip angle
  const hipAngle = angleBetween(midSho, midHip, midKnee);

  // Hip symmetry
  const symmetry = Math.abs(lHip.y - rHip.y) * 100;

  return {
    kneesLocked,
    shouldersBack,
    hipAngle,
    kneeAngle,
    trunkLean,
    symmetry,
    leftWrist: lWrist,
    rightWrist: rWrist,
  };
}

interface DeadliftAnalysisProps {
  frameIndex: number;
  videoName: string;
  totalFrames: number;
}

export function DeadliftAnalysis({
  frameIndex,
  videoName,
  totalFrames,
}: DeadliftAnalysisProps) {
  const [allMetrics, setAllMetrics] = useState<(DeadliftMetrics | null)[]>([]);
  const [loading, setLoading] = useState(true);

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

  const globalValidation = useMemo(() => {
    const valid = allMetrics.filter(Boolean) as DeadliftMetrics[];
    if (!valid.length) return null;

    // Locked knees
    const lockedReached = valid.some((m) => m.kneesLocked);

    // Shoulders back
    const shouldersBackReached = valid.some((m) => m.shouldersBack);

    // Maximum trunk lean
    const maxTrunkLean = Math.max(...valid.map((m) => m.trunkLean));

    // Maximum symmetry
    const maxSymmetry = Math.max(...valid.map((m) => m.symmetry));

    // Minimum hip angle
    const minHipAngle = Math.min(...valid.map((m) => m.hipAngle));

    // Distance between the hands
    let leftHandDist = 0;
    let rightHandDist = 0;
    for (let i = 1; i < valid.length; i++) {
      leftHandDist += dist3D(valid[i].leftWrist, valid[i - 1].leftWrist);
      rightHandDist += dist3D(valid[i].rightWrist, valid[i - 1].rightWrist);
    }
    leftHandDist *= 10;
    rightHandDist *= 10;

    return {
      lockedReached,
      shouldersBackReached,
      maxTrunkLean,
      maxSymmetry,
      minHipAngle,
      leftHandDist,
      rightHandDist,
    };
  }, [allMetrics]);

  const currentMetrics = allMetrics[frameIndex];

  if (loading)
    return <div style={overlayStyle}>Chargement des métriques...</div>;

  return (
    <div style={overlayStyle}>
      <div style={{ fontWeight: "bold", marginBottom: 10, fontSize: 15 }}>
        Analyse soulevé de terre
      </div>

      {globalValidation && (
        <>
          <div style={{ color: "#aaa", fontSize: 11, marginBottom: 6 }}>
            VALIDATION GLOBALE
          </div>
          <Row
            label="Genoux verrouillés"
            value={
              globalValidation.lockedReached ? "✅ Valide" : "❌ Non atteint"
            }
            ok={globalValidation.lockedReached}
          />
          <Row
            label="Épaules en arrière"
            value={
              globalValidation.shouldersBackReached
                ? "✅ Valide"
                : "❌ Non atteint"
            }
            ok={globalValidation.shouldersBackReached}
          />
          <Row
            label="Inclinaison max buste"
            value={`${globalValidation.maxTrunkLean.toFixed(1)}°`}
            ok={globalValidation.maxTrunkLean < 60}
          />
          <Row
            label="Asymétrie max"
            value={`${globalValidation.maxSymmetry.toFixed(1)} mm`}
            ok={globalValidation.maxSymmetry < 20}
          />
          <Row
            label="Angle hanche min"
            value={`${globalValidation.minHipAngle.toFixed(1)}°`}
            ok={true}
          />
          <Row
            label="Main gauche"
            value={`${globalValidation.leftHandDist.toFixed(1)} cm`}
            ok={true}
          />
          <Row
            label="Main droite"
            value={`${globalValidation.rightHandDist.toFixed(1)} cm`}
            ok={true}
          />
        </>
      )}

      {currentMetrics && (
        <>
          <div style={{ color: "#aaa", fontSize: 11, margin: "10px 0 6px" }}>
            FRAME COURANTE
          </div>
          <Row
            label="Angle hanche"
            value={`${currentMetrics.hipAngle.toFixed(1)}°`}
            ok={currentMetrics.hipAngle > 150}
          />
          <Row
            label="Inclinaison buste"
            value={`${currentMetrics.trunkLean.toFixed(1)}°`}
            ok={currentMetrics.trunkLean < 60}
          />
          <Row
            label="Symétrie"
            value={`${currentMetrics.symmetry.toFixed(1)} mm`}
            ok={currentMetrics.symmetry < 20}
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
