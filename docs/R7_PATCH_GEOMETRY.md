# R7 native-grid patch geometry (#28)

R7 now pads the finite south/east edges to whole patches using replicated edge
values, decodes the resulting ceil-grid and crops exactly to original H/W.
It does not drop the last row/column and then bilinearly resize the tendency.
Coordinates are not shifted or resized. Draft feedback uses the same ceil-grid.
V6's CoarseEncoder default remains unchanged; R7 explicitly opts into padding.

For divisible H/W the parameter/state_dict shapes and encoder numerical behavior
are unchanged. On odd grids, behavior intentionally changes to retain trailing
input evidence. Local checkpoints now carry an active-model source digest and
refuse a different model implementation during exact resume/evaluation. Old
pre-fix checkpoints must be evaluated with their recorded code, not silently
compared under the new geometry. Re-training/re-evaluation is necessary for new
scientific comparisons; no existing checkpoint files are modified.

Periodic R7 longitude is supported only when width is divisible by
patch_size*window_size; unsupported partial periodic windows fail explicitly.
This fixes geometry, not model resolution skill. Real native 0.25-degree training
fields are still required for a native 0.25-degree scientific claim.
