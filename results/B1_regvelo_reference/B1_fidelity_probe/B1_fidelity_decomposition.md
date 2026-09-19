# B1 L2/L3 fidelity decomposition probe

**Status:** trained
**Epochs:** 5
**Train runtime:** 27.72 s
**Velocity runtime:** 3.59 s
**Peak GPU:** 4.933 GB

## Parameter inventory

- `v_encoder.beta_mean_unconstr`: shape=[2000], raw range=[0.0511094331741333, 0.9488695859909058], post-transform=softplus + clamp(0,50), post range=[0.7190283536911011, 1.2761414051055908], role=N/A
- `v_encoder.gamma_mean_unconstr`: shape=[2000], raw range=[-1.482187032699585, -0.7700831890106201], post-transform=softplus + clamp(0,50), post range=[0.20468656718730927, 0.38047173619270325], role=N/A
- `alpha_1_unconstr`: shape=[2000], raw range=[0.0, 0.0], post-transform=N/A, post range=N/A, role=max transcription rate / bias term in generative model
- `alpha_unconstr`: shape=[2000], raw range=[-inf, 127.9808349609375], post-transform=N/A, post range=N/A, role=global transcription rate offset
- `decoder.px_rho_decoder.0.bias`: shape=[2000], raw range=[-0.17681211233139038, 0.16001231968402863], post-transform=N/A, post range=N/A, role=per-gene latent-time decoder bias
- `v_encoder.fc1.bias`: shape=[2000], raw range=[-0.2257206290960312, 0.19697584211826324], post-transform=N/A, post range=N/A, role=per-gene rate encoder bias

## Mapping feasibility assessment

**Clean L2/L3 decomposition feasible:** False

### API paths inspected

- `REGVELOVI.setup_anndata`
- `REGVELOVI(adata, W=W_tensor, soft_constraint=True)`
- `model.module.named_parameters()`
- `model.module._get_rates() -> (gamma, beta, alpha_1)`
- `model.module.v_encoder.beta_mean_unconstr / gamma_mean_unconstr`
- `model.module.alpha_1_unconstr`
- `model.module.decoder(decoder_input) -> px_rho`
- `model.module.get_px(px_rho, scale, gamma, beta, alpha_1)`
- `model.get_velocity(adata, return_numpy=True)`
- `SAVI run_l3_only.load_l2_checkpoint expects beta.npy/gamma.npy/bias.npy/t.npy/gate.npy/W.npz/metadata.json`

### Blocking reasons

- RegVelo inference_outputs beta/gamma are per-cell (std across cells: beta=5.866e-02, gamma=1.261e-02), not a single per-gene vector; SAVI L3 expects per-gene beta/gamma.
- RegVelo generative path predicts per-cell/gene latent-time modulation px_rho (std=9.156e-02); SAVI L3 has no equivalent latent-time decoder input.
- RegVelo velocity = beta * mean_u(ind_t) - gamma * mean_s(ind_t) where (mean_u, mean_s) come from _get_induction_unspliced_spliced; SAVI L3 propagates observed Ms/Mu through a Picard segment-wise ODE with GRN-derived bias. The two propagation steps are structurally different.

### Per-cell rate statistics

- beta_std_per_gene_mean: 5.866e-02
- gamma_std_per_gene_mean: 1.261e-02
- px_rho_std_per_gene_mean: 9.156e-02

**Verdict:** A direct injection of RegVelo kinetic parameters into SAVI L3 is not supported by the public API. RegVelo couples kinetic inference with a per-cell latent-time decoder and a closed-form induction-time solution, whereas SAVI L3 expects static per-gene beta/gamma plus a GRN-derived bias. Fidelity decomposition therefore remains blocked.