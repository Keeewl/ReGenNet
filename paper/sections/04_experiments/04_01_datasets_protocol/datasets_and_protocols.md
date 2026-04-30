## Datasets

We primarily evaluate HiReact on the Inter-X dataset. Inter-X is a large-scale human-human interaction dataset covering 40 daily interaction categories, with about 11K interaction sequences and more than 8.1M motion frames. The dataset provides both full-body motion and fine-grained hand motion, together with interaction-order annotations, which makes it well suited for actor-reactor asymmetric human reaction generation. Since this work focuses on fine-grained reaction generation, especially hand responses, local contact, and interaction geometry, Inter-X serves as the main dataset for both Stage1 and Stage2 evaluations.

For Stage2 evaluation, we further construct a contact-focused subset from Inter-X by selecting 15 contact-rich action categories. This subset contains denser and more explicit hand-centric interaction patterns, and is therefore more suitable for evaluating local contact quality, contact-relevant window refinement, and fine-grained interaction modeling. Unless otherwise specified, all Stage2 results reported in the main paper are based on this 15-contact-rich subset.

In addition, we use CHI3D as a supplementary dataset and report its results in the appendix. CHI3D contains more challenging close-contact interaction scenes and is used to further examine the generalization ability and robustness of our method under stronger occlusion and denser contact conditions.


## Evaluation Protocols

We adopt two complementary evaluation protocols. For STGCN-based motion distribution evaluation, we compare generated motions in canonical space to assess overall motion quality and distributional similarity. For contact-related evaluation, we perform comparison in restored real-world space, where the original body shape and gender information of the interacting subjects are restored, so that hand-target contact and local interaction geometry can be evaluated more faithfully.

The main Stage1 experiments are conducted on the full Inter-X dataset under the online, unconstrained setting, where actor motion is the only conditioning signal and no additional text condition is used. Unless otherwise specified, the main Stage1 results are reported in canonical space and evaluated with STGCN-based metrics.

The main Stage2 experiments are conducted on the 15-contact-rich subset using a full-sequence evaluation protocol. Specifically, STGCN metrics for Stage2 are still evaluated in canonical space, while contact metrics are evaluated in restored real-world space. This protocol allows us to separately assess how Stage2 affects the overall motion quality of the refined sequence and the local contact accuracy.


## Evaluation Metrics

For Stage1, we use STGCN-based motion distribution metrics to evaluate the overall quality of generated reactions, including Fr\'echet Inception Distance (FID), Accuracy (Acc.), Diversity (Div.), and Multimodality (Multimod.). These metrics respectively measure distributional similarity, action recognizability, motion diversity, and the multimodal nature of conditional generation. 

For Stage2, we report both motion distribution metrics and contact quality metrics. The STGCN block still includes FID, Acc., Div., and Multimod., which are used to assess the overall motion quality of the refined sequence. The contact block contains four core metrics: Contact F1, Recall, Contact Distance, and Contact Ratio. Contact F1 measures overall contact prediction quality, Recall reflects the ability to recover true contact events, Contact Distance evaluates the geometric hand-target proximity, and Contact Ratio describes the proportion of contact frames over the full sequence. Together, these metrics allow us to evaluate Stage2 from both the global motion-quality perspective and the local contact-accuracy perspective.
