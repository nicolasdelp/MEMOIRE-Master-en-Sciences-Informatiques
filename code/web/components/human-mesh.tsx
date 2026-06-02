import { Center, OrbitControls } from "@react-three/drei";
import { useLoader, useThree } from "@react-three/fiber";
import { button, folder, useControls } from "leva";
import { useEffect, useRef, useState } from "react";
import { DoubleSide } from "three";
import { PLYLoader } from "three/examples/jsm/Addons.js";

export function HumanMesh({urls}: {urls: string[]}) {
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const controlsRef = useRef<any>(null);
  const { camera } = useThree();

  const setCameraView = (view:  'front' | 'left' | 'back' | 'right') => {
    if (!controlsRef.current) return;

    const views = {
      left: { pos: [5, 1, 0], target: [0, 0, 0] }, // Vue de Gauche
      front: { pos: [0, 1, 5], target: [0, 0, 0] }, // Vue de face
      back: { pos: [0, 1, -5], target: [0, 0, 0] }, // Vue de dos
      right: { pos: [-5, 1, 0], target: [0, 0, 0] }, // Vue de Droite
    };

    const config = views[view];
    camera.position.set(config.pos[0], config.pos[1], config.pos[2]);
    controlsRef.current.target.set(config.target[0], config.target[1], config.target[2]);
    controlsRef.current.update();
  };

  const [, set] = useControls(() => ({
    frame: {
      value: 0,
      min: 0,
      max: urls.length - 1,
      step: 1,
      onChange: (v) => setCurrentFrame(v),
    },
    'play/pause': button(() => setIsPlaying((prev) => !prev)),
    'Camera Views': folder({
      'Front View': button(() => setCameraView('front')),
      'Back View': button(() => setCameraView('back')),
      'Left View': button(() => setCameraView('left')),
      'Right View': button(() => setCameraView('right')),
    }, { collapsed: true })
  }), [urls]);

  useEffect(() => {
    set({ frame: currentFrame });
  }, [currentFrame, set]);

  const geometries = useLoader(PLYLoader, urls);

  useEffect(() => {
    geometries.forEach((g) => g.computeVertexNormals());
  }, [geometries]);

  useEffect(() => {
    let interval: any;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentFrame((prev) => (prev + 1) % urls.length);
      }, 16); // +- 60 FPS
    }
    return () => clearInterval(interval);
  }, [isPlaying, urls.length]);

  return (
    <>
      <OrbitControls ref={controlsRef} makeDefault enablePan={false} />
      <Center top>
        <mesh castShadow geometry={geometries[currentFrame]}>
          <meshStandardMaterial 
              color="#FEC3AC"
              metalness={0} 
              roughness={0.6} 
              side={DoubleSide} 
          />
        </mesh>
      </Center>
    </>
  );
}