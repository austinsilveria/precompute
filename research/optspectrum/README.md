## A Power Spectrum Power Law in Transformer LM Latent Space

Sander Dieleman's [Diffusion is spectral autoregression blog post](https://sander.ai/2024/09/02/spectral-autoregression.html) observes that natural images have an approximate power law in their power spectrum--i.e. averaged over all spatial directions in the image (rotating a ray from the origin to the edge of the image 360 degrees around), pixel brightnesses are dominated by low frequencies (slow brightness change/coarse structure) with contributions from higher frequencies (fast brightness change/granular structure) dropping off exponentially according to a smooth power law.

In the following image, we're looking at a log-log plot of the radially averaged power spectrum of the image (red line), the noise power spectrum (blue line), and the image + noise (green line).

<p align="center">
  <img src="https://github.com/user-attachments/assets/1449516e-374f-4f32-a2f8-54972c0bbab0" />
  </p>
  <p align="center">
    <a href="https://sander.ai/2024/09/02/spectral-autoregression.html">Fig 1. Diffusion is spectral autoregression blog post</a>
  </p>

So, we see that the image has a power law, the noise has uniform magnitude across all frequencies, and adding noise drowns out frequencies with lower magnitude (since the frequency powers vary across orders of magnitude).

This means that in reverse diffusion, low frequencies are added to the image first, corresponding to observed behavior of diffusion models predicting coarse structure first before gradually refining it.

If this is the case for images, how can we apply this analysis to language?

In the following graphs, we're looking at the power spectrum of each hidden state dimension across sequence length, averaged across all hidden state dimensions and a batch of 8 sequences (768 hidden state dimensions for OPT-125m).

```python
# Tok Embeddings, Tok + Pos Embeddings, 12 layers
# batch - 8, sequence length - 2048, hidden dimension - 768
hidden_states: [14, 8, 2048, 768]
x_axis = torch.fft.rfftfreq(hidden_states.shape[2])
fft = torch.fft.rfft(hidden_states, dim=2)
y_axis = torch.mean(fft**2, dim=(1, 3))
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/5c3c621c-dc41-488a-a87d-d6794b634ae6" />
</p>
<p align="center">
    Fig 2. The intermediate hidden states of a pretrained OPT-125m.
  </p>

Ok, so we're seeing uniform power across frequency just after the initial token embeddings (bottom blue line). This makes sense because the each token representation after the initial embeddings is not a function of context, and as a result, the representations across the sequence length dimension are changing more abruptly. 

Then, we're seeing an increase in low frequencies after the position embeddings get added. And then we see the power law progressively straighten out as the representation passes through the 12 transformer blocks.

One difference from image's power spectrum graph is the range: The image's power law spans ~5 OOMs whereas the transformer output's power law only spans ~1 OOM in same frequency interval. So, although we do see a power law in the transformer output, the representation is not as dominated by low frequencies as the image.

What does this look like for a randomly initialized OPT-125m?

<p align="center">
  <img src="https://github.com/user-attachments/assets/f310b3a8-79a1-48b1-83b6-28bd4d74947c" />
</p>
<p align="center">
    Fig 3. The intermediate hidden states of a randomly initialized OPT-125m.
  </p>

At initialization the position embeddings are flat, but after training the position embeddings have come to more heavily favor low frequencies, perhaps adding an initial long range structure to the representation. And the hidden layers have learned to "straighten" out the power law, balancing the contribution of the frequency spectrum according to a smooth power law.

Ok, so maybe we can interpret this as the transformer learning to smooth out the sharpness of language's latent space structure by aggregating sequence context.

Why is the model attracted to parameters that produce these smooth representations? If we see these spectral properties in images, slightly proccessed audio clips (see Sander's blog post), and transformer latent space outputs, is there something fundamental efficient information representations here? Do other sequence models such as the [Linear Recurrent Unit](https://arxiv.org/abs/2303.06349) or [Mamba](https://arxiv.org/abs/2312.00752) produce outputs with these properties as well?

Or is there an artifact of the transformer architecture that biases training towards parameters that produce these representations?

<p align="center">
  <img src="https://github.com/user-attachments/assets/387b787c-a12c-48c4-9c55-9cdeab37b4e7" />
</p>
<p align="center">
    Fig 4. The attention residuals of an untrained OPT-125m.
  </p>

Now, this isn't a completely straight line, but it's pretty close (layer 1 is our outlier friend). Is the initial behavior of the attention mechanism biasing training towards producing outputs with a smooth power law?

  <p align="center">
  <img src="https://github.com/user-attachments/assets/71e9f2c4-3597-419b-af25-d37f1c40346e" />
</p>
<p align="center">
    Fig 5. The MLP residuals of an untrained OPT-125m.
  </p>

And for completeness, the initial MLP residuals are pretty close to the inital hidden states.

The power spectrums of the trained attention and MLP residuals are also informative--since each residual is of smaller magnitude and added to the representation, we can interpret them as "pushing" the structure of the representation in a certain direction.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ea401d5b-57a4-471f-a3dc-b7023ab0b298" />
</p>
<p align="center">
    Fig 6. The attention residuals of a pretrained OPT-125m.
  </p>

The first three layers (bottom lines) have some spikes, and then the representations smoothen out until the final layer has another low frequency spike.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bb3029ec-c5d1-422a-a6e0-ae5460ac9f2e" />
</p>
<p align="center">
    Fig 7. The MLP residuals of a pretrained OPT-125m.
  </p>

The first layer here has a big spike again, and then things smoothen out with no spike at the end.

The early layers of both residuals look quite different than the rest of the layers--since these are the first residuals that get to add information as a function of sequence context, perhaps disproportionately adding smooth long range structure to the representation is the most useful strategy right at the beginning, and then later layers can more evenly refine features across the spectrum.

This seems to align with [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) findings that "the inputs are immediately converted to a very different representation, which is smoothly refined into the final prediction."

So if our final hidden states have a smooth power law across the power spectrum, does this mean diffusion over this representation will work well?

[Self-Conditioned Embedding Diffusion for Text Generation](https://arxiv.org/abs/2211.04236) trains a bidirectional attention transformer with a diffusion objective over word embeddings, but they use the initial token embedding space: "To generate word embeddings, we train a BERT model of fixed size (150m parameters) and feature dimension dmodel = 896. The diffusion space is defined by the initial lookup table of this BERT model."

From Figure 2, we see uniform power across frequency in the representation directly after the token embeddings in OPT-125m. As Dieleman's [blog post](https://sander.ai/2024/09/02/spectral-autoregression.html) asks if image diffusion performs well because of the power spectrum power law of natural images, would diffusion over the transformer's final hidden states, with their power spectrum power law representation, perform better than diffusion over the intial token embedding space?

This would require a pretrained language model to generate representations, but studying the differences between diffusion over these two spaces may be fruitful.

#### Open Questions

1. Is the attention mechanism biasing training to produce final hidden states with a power spectrum power law? If not, why do transformer language model final hidden states have this clean power law?
2. Can we use diffusion over the final hidden states to non-autoregressively generate language in a coarse to granular fashion, like image diffusion models?
3. Are there other early layer behaviors that align with our findings of them disproportionately adding low frequencies to the representation?
