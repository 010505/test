import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { OFFICIAL_GESTURES, gestureDisplayName } from "../js/gestures.js";
import { GestureSegmenter, LandmarkSmoother, PredictionStabilizer, diagnosticGesture, motionEnergy, normalizeCanonical, toCanonical22 } from "../js/skeleton.js";

const hand = Array.from({ length: 21 }, (_, i) => ({ x: i * .01, y: i * .02, z: i * -.003 }));

test("maps 21 MediaPipe landmarks to 22 canonical nodes", () => {
  const graph = toCanonical22(hand);
  assert.equal(graph.length, 22);
  assert.ok(graph[0].every(value => Math.abs(value) === 0));
  assert.deepEqual(graph[2], [.01, .02, -.003]);
});

test("normalization places wrist at the origin and remains finite", () => {
  const normalized = normalizeCanonical(toCanonical22(hand));
  assert.ok(normalized[0].every(value => Math.abs(value) === 0));
  assert.ok(normalized.flat().every(Number.isFinite));
});

test("motion energy is zero for identical frames", () => {
  const graph = normalizeCanonical(toCanonical22(hand));
  assert.equal(motionEnergy(graph, graph), 0);
});

test("smoother returns defensive copies", () => {
  const smoother = new LandmarkSmoother(.5);
  const result = smoother.update([[0, 0, 0]]);
  result[0][0] = 99;
  assert.equal(smoother.update([[0, 0, 0]])[0][0], 0);
});

test("diagnostic classifier returns bounded confidence", () => {
  const graph = normalizeCanonical(toCanonical22(hand));
  const result = diagnosticGesture(graph);
  assert.ok(result.label);
  assert.ok(result.confidence >= 0 && result.confidence <= 1);
});

test("gesture segmenter starts on motion and ends after stillness", () => {
  const segmenter = new GestureSegmenter({ startThreshold: .2, endThreshold: .1, startFrames: 2, endFrames: 2, minFrames: 4, maxFrames: 10, preRollFrames: 2, cooldownFrames: 2 });
  const frame = [[0, 0, 0]];
  assert.equal(segmenter.update(frame, .05).state, "idle");
  assert.equal(segmenter.update(frame, .3).started, false);
  const started = segmenter.update(frame, .3);
  assert.equal(started.started, true);
  assert.equal(started.progress, 2);
  segmenter.update(frame, .3);
  segmenter.update(frame, .05);
  const ended = segmenter.update(frame, .05);
  assert.equal(ended.state, "cooldown");
  assert.equal(ended.reason, "still");
  assert.equal(ended.completed.length, 5);
});

test("prediction stabilizer requires consecutive confident agreement", () => {
  const stabilizer = new PredictionStabilizer({ confidenceThreshold: .5, minVotes: 2 });
  const prediction = (label, confidence) => ({ label, confidence, scores: { [label]: confidence } });
  assert.equal(stabilizer.add(prediction("tap", .8)).status, "pending");
  const stable = stabilizer.add(prediction("tap", .7));
  assert.equal(stable.status, "stable");
  assert.equal(stable.label, "tap");

  stabilizer.reset();
  stabilizer.add(prediction("tap", .8));
  const unknown = stabilizer.add(prediction("shake", .7), true);
  assert.equal(unknown.status, "unknown");
});

test("official gesture catalogue has 14 unique labels and reference animations", () => {
  const labels = OFFICIAL_GESTURES.map(gesture => gesture.label);
  assert.equal(labels.length, 14);
  assert.equal(new Set(labels).size, 14);
  labels.forEach(label => assert.ok(existsSync(new URL(`../assets/references/${label}.gif`, import.meta.url))));
  assert.match(gestureDisplayName("tap"), /点按/);
});
