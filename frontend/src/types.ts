export type Box = [number, number, number, number];

export interface Detection {
  label: string;
  confidence: number;
  box: Box;
  track_id?: number | null;
}

export interface GroundTruth {
  label: string;
  box: Box;
}

export interface TestImage {
  id: string;
  path: string;
  source: string;
  scene: string;
  category: string;
  width: number;
  height: number;
  ground_truth: GroundTruth[];
  detections: Detection[];
  status: string;
  false_positives: number;
  false_negatives: number;
}

export interface CameraInfo {
  id: string;
  name: string;
  uri: string;
  enabled: boolean;
  status: string;
  fps: number;
  frames: number;
  alarm: Record<string, unknown>;
}
