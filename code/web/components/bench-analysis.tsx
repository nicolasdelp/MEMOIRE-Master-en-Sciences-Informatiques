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

interface BenchMetrics {
  elbowAngle: number;
  armsLocked: boolean;
  wristSymmetry: number;
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

function computeMetrics(kp: Keypoint3D[]): BenchMetrics {
  const lSho = kp[JOINTS.LEFT_SHOULDER];
  const rSho = kp[JOINTS.RIGHT_SHOULDER];
  const lElbow = kp[JOINTS.LEFT_ELBOW];
  const rElbow = kp[JOINTS.RIGHT_ELBOW];
  const lWrist = kp[JOINTS.LEFT_WRIST];
  const rWrist = kp[JOINTS.RIGHT_WRIST];

  // Elbow angle
  const lElbowAngle = angleBetween(lSho, lElbow, lWrist);
  const rElbowAngle = angleBetween(rSho, rElbow, rWrist);
  const elbowAngle = (lElbowAngle + rElbowAngle) / 2;

  // Arms locked
  const armsLocked = lElbowAngle > 160 && rElbowAngle > 160;

  // Wrist symmetry
  const wristSymmetry = Math.abs(lWrist.y - rWrist.y) * 100;

  return {
    elbowAngle,
    armsLocked,
    wristSymmetry,
    leftWrist: lWrist,
    rightWrist: rWrist,
  };
}

interface BenchAnalysisProps {
  frameIndex: number;
  videoName: string;
  totalFrames: number;
}

export function BenchAnalysis({
  frameIndex,
  videoName,
  totalFrames,
}: BenchAnalysisProps) {
  const [allMetrics, setAllMetrics] = useState<(BenchMetrics | null)[]>([]);
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
    const valid = allMetrics.filter(Boolean) as BenchMetrics[];
    if (!valid.length) return null;

    // True if AT LEAST ONE frame has locked elbows
    const lockedReached = valid.some((m) => m.armsLocked);

    // Minimum bend angle
    const minElbowAngle = Math.min(...valid.map((m) => m.elbowAngle));

    // Maximum wrist symmetry
    const maxWristSymmetry = Math.max(...valid.map((m) => m.wristSymmetry));

    // Hand distance
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
      minElbowAngle,
      maxWristSymmetry,
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
        Analyse développé couché
      </div>

      {globalValidation && (
        <>
          <div style={{ color: "#aaa", fontSize: 11, marginBottom: 6 }}>
            VALIDATION GLOBALE
          </div>
          <Row
            label="Verrouillage coudes"
            value={
              globalValidation.lockedReached ? "✅ Valide" : "❌ Non atteint"
            }
            ok={globalValidation.lockedReached}
          />
          <Row
            label="Angle coude min"
            value={`${globalValidation.minElbowAngle.toFixed(1)}°`}
            ok={globalValidation.minElbowAngle < 90}
          />
          <Row
            label="Symétrie poignets"
            value={`${globalValidation.maxWristSymmetry.toFixed(1)} mm`}
            ok={globalValidation.maxWristSymmetry < 20}
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
            label="Angle coude"
            value={`${currentMetrics.elbowAngle.toFixed(1)}°`}
            ok={currentMetrics.elbowAngle > 160}
          />
          <Row
            label="Symétrie poignets"
            value={`${currentMetrics.wristSymmetry.toFixed(1)} mm`}
            ok={currentMetrics.wristSymmetry < 20}
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
