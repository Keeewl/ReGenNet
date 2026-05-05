
\begin{table*}[t]
\centering
\caption{
Quantitative comparison on the contact-rich Inter-X subset.
$\downarrow$ indicates lower is better, $\uparrow$ indicates higher is better, and $\rightarrow$ indicates closer to real motion is better.
HiReact* denotes the Stage I coarse reaction before local refinement.
}
\label{tab:contact_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccccc}
\toprule
Method
& FID$\downarrow$
& Acc.$\rightarrow$
& Div.$\rightarrow$
& MMod.$\rightarrow$
& Contact F1$\uparrow$
& Recall$\uparrow$
& Contact Dist.$\downarrow$
& Contact Ratio$\uparrow$ \\
\midrule
Real
& -
& 0.986
& 21.56
& 4.76
& -
& -
& 0.02
& 0.490 \\
\midrule
AGRoL
& 7.88
& 0.860
& 19.78
& 4.93
& 0.237
& 0.181
& 0.22
& 0.339 \\
MDM
& 6.20
& 0.869
& 19.61
& 4.93
& 0.273
& 0.204
& 0.23
& 0.307 \\
MDM-GRU
& 4.43
& 0.915
& 20.18
& 4.82
& 0.344
& 0.261
& 0.21
& 0.316 \\
ReGenNet
& 1.22
& 0.968
& 20.90
& \textbf{4.81}
& 0.672
& 0.614
& 0.05
& 0.439 \\
\midrule
HiReact*
& \underline{0.47}
& \underline{0.978}
& \underline{21.20}
& 4.84
& \underline{0.782}
& \underline{0.730}
& \underline{0.03}
& \underline{0.442} \\
HiReact
& \textbf{0.30}
& \textbf{0.978}
& \textbf{21.27}
& \underline{4.82}
& \textbf{0.795}
& \textbf{0.749}
& \textbf{0.03}
& \textbf{0.448} \\
\bottomrule
\end{tabular}
}
\end{table*}