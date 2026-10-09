import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';

const src = 'D:/科研/小论文/架构图/KineTalk_学术架构图_新版.pptx';
const out = 'D:/科研/小论文/架构图/ppt_refactor_build/KineTalk_学术架构图_学术命名版_draft.pptx';
const p = await PresentationFile.importPptx(await FileBlob.load(src));

const replacements = [
  ['(a) KineTalk overview', '(a) KineTalk overview'],
  ['(b) Residual Flow Decoder', '(b) Conditional Residual Motion Decoder'],
  ['Target speech\nA', 'Input Speech'],
  ['Speech\nEncoder', 'Articulation\nEncoder'],
  ['B₀ mapping\n(frozen)', 'Frozen Articulation\nBase'],
  ['Audio Content Mapping', 'Articulation'],
  ['Audio Emotion Encoding', 'Emotion'],
  ['Audio\nEncoder', 'Audio Emotion\nEncoder'],
  ['Temporal Audio', 'Temporal Acoustic\nFeatures'],
  ['Neutral reference\naudio + motion\n{Aᵣ, Mᵣ}', 'Neutral Reference\nPair'],
  ['Mᵣ − B₀(Cᵣ)', 'Reference\nMotion'],
  ['Indentity\nEncoder', 'Reference Motion\nEncoder'],
  ['Statistics', 'Reference\nPooling'],
  ['Neutral reference identity', 'Personalized Motion'],
  ['Residual flow\ndecoder Fθ', 'Conditional Residual\nMotion Decoder'],
  ['Non-mouth\nmask S', 'Mouth\nProtection'],
  ['Identity offset\nb_id', 'Personalized\nOffset'],
  ['Global affect distillation (training only)', 'Emotion Alignment (training only)'],
  ['r* = M − b₀(C) − b_id', 'Training Motion'],
  ['Motion teacher', 'Motion Emotion\nEncoder'],
  ['MSE(gₐ, sg(gₘ))\n(global semantics)', 'Emotion Alignment\nLoss'],
  ['Hctx = Pₕ(h₀) + Pᵤ(uₐ)\ng = P_g(gₐ) + P_z(z_id) + P_τ(τ)', 'Condition inputs'],
  ['Pₓ(xτ) + Hctx', 'Noisy residual +\ncontext'],
  ['Global modulation\ngₐ + z_id + τ', 'Emotion and time\nmodulation'],
  ['K,V = Hctx', 'Temporal context'],
  ['vθ(xτ, τ)', 'Velocity field'],
  ['Flow integration\nε → r̂', 'Flow integration'],
  ['M̂ = b₀(C) + b_id + S ⊙ r̂', 'Residual facial\nmotion'],
  ['S mouth/jaw = 0', 'Mouth protection'],
  ['Figure 1. KineTalk preserves articulation with a frozen Stage-1 base while audio affect conditions drive stochastic non-mouth residual dynamics.', 'Figure 1. KineTalk separates articulation, personalized motion, and emotion, then synthesizes residual facial motion with a conditional flow decoder.'],
];

const snap = await p.inspect({kind:'textbox,notes', maxChars:50000});
const records = snap.ndjson.split(/\n/).filter(Boolean).map(x => JSON.parse(x));
let changed = [];
for (const rec of records) {
  if (!rec.id || rec.kind !== 'textbox') continue;
  const hit = replacements.find(([from]) => rec.text === from);
  if (!hit) continue;
  const target = p.resolve(rec.id);
  target.text.replace(hit[0], hit[1]);
  changed.push({id:rec.id, from:hit[0], to:hit[1]});
}

const note = p.resolve('nt/y90nupkv');
note.textFrame.setText('KineTalk uses a frozen articulation base for speech-synchronized mouth motion, a neutral reference encoder for personalized motion, and an audio emotion encoder that produces global emotion and temporal acoustic features. A motion emotion encoder is used only during training for emotion alignment. The conditional residual motion decoder combines articulation features, personalized motion, emotion, temporal acoustic context, and random noise to synthesize ARKit52 facial animation. The main path excludes the legacy Stage5 upper-flow branch and does not require a target motion sequence or emotion label at inference.');

await fs.mkdir(path.dirname(out), {recursive:true});
await (await PresentationFile.exportPptx(p)).save(out);
console.log(JSON.stringify({out, changedCount:changed.length, changed}, null, 2));
