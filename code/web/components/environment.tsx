import { Environment as ThreeEnvironment } from '@react-three/drei'

export function Environment() {
  return <ThreeEnvironment preset={"dawn"} background blur={1} />
}