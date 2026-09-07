# LVQ-CF: Prototype-Based Counterfactual Explainability

Implementation of LVQ-CF, a prototype-based counterfactual explanation method, applicable to any scikit-learn-compatible supervised classification model.

## Description

This project proposes an extension of the Bergamin & Aiolli approach, replacing k-medoids prototypes with LVQ prototypes positioned near the decision boundary.

## Comparison Methods Used

- Wachter et al.
- DiCE
- CFProto ([Van Looveren & Klaise](https://github.com/SeldonIO/alibi))
- NNContrastive (AIX360)
- Bergamin & Aiolli — [link to their repo]
- Artelt & Hammer

## Datasets

banknote, boston, breast_cancer, iris, magic, moons, wine
