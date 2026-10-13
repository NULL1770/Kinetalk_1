# 风格论证与创新点验收

## 风格定义

Speaking style is limited to cross-utterance and cross-emotion articulation habits: mouth aperture scale, mouth width/bias, pouting and lip/jaw coordination. It is not query phonetic content or query emotion.

## Mimic 对照

- Mimic uses mouth opening/closing amplitude and pouting as style examples.
- Mimic evaluates lip dynamics deviation (LDD) and style cosine similarity (SCS) with an independent motion classifier.
- Mimic removes style classifier, inverse identity classifier, content contrastive loss and latent cycles in ablations.
- The current v5 audit has no independent SCS classifier, official HDTF mesh split or sealed test, so it is proxy evidence only.

## 当前证据

- Same-person cross-emotion reference: aperture correlation 0.99849 and velocity cosine 0.99656.
- Cross-person reference: aperture correlation 0.90939, velocity cosine 0.84034, amplitude ratio 1.23727 and mouth-width absolute change 2.15616 mm.
- Matched cross-person style direction: target descriptor error changes from 0.47667 to 0.36295 for frozen start, with direction cosine 0.61283.
- Automatic development emotion probes are .688/.625/.641/.672 with mouth MAE 0.036635; upper-face sampled amplitude is 0.0480 versus GT 0.0826.

|候选贡献|当前状态|缺口|
|---|---|---|
|reference-conditioned articulation style|partial proxy evidence|independent SCS/LDD, unseen-speaker reference swap and component ablation|
|teacher/student affect trend plus stochastic dynamics|not established|teacher/audio/student ablation, multi-sample coverage and intensity-stratified event metrics|
|joint content-affect-style decoder|partial development evidence|counterfactual swaps with lip closure, sync and emotion preservation on a sealed split|

## 不能写成已成立的结论

- Three original contributions are not established by the current development audits.
- Complete emotion/style disentanglement is not established; reference swaps retain emotion leakage.
- Teacher deployment improvement is not established; the new StateFlow is disabled in the frozen release.
- Dynamic intensity is under-recovered and Angry upper-face tension remains weak.
