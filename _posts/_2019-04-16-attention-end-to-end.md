---
title: "Attention based end-to-end architectures for Automatic Speech Recognition"
# categories: [speech, attention]
# tags: [espnet, speech]
excerpt: "Attention based end-to-end architectures for Automatic Speech Recognition"
header:
  overlay_color: "#033b52"
  teaser: /assets/images/code_teaser.png

toc: true
# toc_sticky: true
classes: wide
---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
TL;DR - A detailed discussion of the components of end-to-end models with neural Attention for ASR. Includes feature extraction, encoder design, CTC, Attention, RNN Decoder, RNN LM, Embeddings, Multitask learning, Label smoothing.

## Feature extraction
As with any other problem involving machine learning, we can do feature extraction in many ways. For ASR, people have used many types of features including MFCC, filter bank, pitch, plp, spectrogram etc.

There are many libraries out there which support a subset of these features. In most of the cases, people tend to lean towards kaldi-asr for it's extensive list of options.

If we choose to extract filter bank coefficients for training our end-to-end model, we could do it the kaldi way and then feed the features to our python code using the kaldiio library.

In any case, we have to find a way to feed the feature vectors to our model in an efficient way. ESPnet uses kaldiio to read ark files from the kaldi feature dump, convert them to torch tensors and feed them to our models.

Our input data might look like this if we go the kaldi way. This json file contains the location, shape, length of both inputs and outputs.

``` json
{
  "MKAM0_SI1316": {
              "input": [
                  {
                      "feat": "/home/shree/espnet/egs/timit/asr1/dump/train_nodev/deltafalse/feats.6.ark:149596",
                      "name": "input1",
                      "shape": [
                          287,
                          26
                      ]
                  }
              ],
              "output": [
                  {
                      "name": "target1",
                      "shape": [
                          37,
                          42
                      ],
                      "text": "sil s sil p ey s sil p r ow sil b z hh eh v y iy l d ih sil d l ih dx l ih n f er m ey sh ih n sil",
                      "token": "<space> s <space> p ey s <space> p r ow <space> b z hh eh v y iy l d ih <space> d l ih dx l ih n f er m ey sh ih n <space>",
                      "tokenid": "2 31 2 29 15 31 2 29 30 27 2 8 40 18 13 37 39 20 23 10 19 2 10 23 19 12 23 19 25 16 14 24 15 32 19 25 2"
                  }
              ],
              "utt2spk": "MKAM0"
          }
}
```
## Encoder
There are many types of encoders available. In the ESPnet toolkit, we will find these lstm, blstm, lstmp, blstmp, vgglstmp, vggblstmp, vgglstm, vggblstm, gru, bgru, grup, bgrup, vgggrup, vggbgrup, vgggru, vggbgru.

The nomenclature could be confusing. So here's how to interpret them:
- LSTM - Long short-term memory
- GRU - Gated Recurrent Unit
- B + LSTM/GRU - Bidirectional network
- LSTM/GRU + P - Networks with a projection layer (bottlenetck) after every RNN layer
- VGG - Oxford's Visual Geometry Group
- VGG + LSTM/GRU - VGGNet like (2 layer) network preceding the RNNs

## CTC
## Attention Decoder
## Output embeddings
## Loss functions
## Multitask learning
## Regularization techniques