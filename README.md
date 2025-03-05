# Urinary Metabolomics and Machine Learning: A Novel Diagnostic and Grading Method for Canine Mammary Tumors
Authorship TBD
ADD IN TARGET JOURNAL

# Latex
\documentclass{article}
\usepackage{graphicx}

\title{Urinary Metabolomics and Machine Learning: A Novel Diagnostic and Grading Method for Canine Mammary Tumors}
\author{Asher T. Wang, Songyeon Lee, Byung-Joon Seung, Karin U. Sorenmo} (Authorship TBD)
\publication{This paper has been submitted for publication in Veterinary and Comparative Oncology Journal}
\date{\[]}

\begin{document}

\maketitle

\section{Introduction}

This repository contains the code and LaTeX files used to generate the results for our paper, \textit{Urinary Metabolomics and Machine Learning: A Novel Diagnostic and Grading Method for Canine Mammary Tumors}. This study introduces a machine learning (ML)-driven diagnostic model leveraging urinary metabolite profiling to classify malignant masses and assist in tumor grading. Using proton nuclear magnetic resonance (NMR) spectroscopy, we analyze 55 tumor-specific urinary metabolites from 156 dogs to develop accurate, non-invasive diagnostic tools. 

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{example_result.png}
    \caption{Example figure showcasing the main result: Kaplan-Meier survival analysis of metabolic tumor grades (MG1, MG2, and MG3). MG3 tumors exhibit significantly poorer survival outcomes compared to MG1/2, highlighting the prognostic value of urinary metabolomics.}
    \label{fig:main_result}
\end{figure}

\section{Abstract}

\textbf{Introduction:} []

\textbf{Methods:} []
\textbf{Results:} []

\textbf{Conclusion:} []

\section{Software Implementation}

The analysis was conducted using Python within Jupyter notebooks. The key dependencies include:

\begin{itemize}
    \item \textbf{pandas, numpy} -- Data preprocessing and manipulation
    \item \textbf{scikit-learn} -- Machine learning model training
    \item \textbf{matplotlib, seaborn} -- Data visualization
    \item \textbf{lifelines} -- Kaplan-Meier survival analysis
\end{itemize}

The repository includes:
\begin{itemize}
    \item \texttt{notebooks/} -- Jupyter notebooks for data preprocessing, feature engineering, model training, and evaluation
    \item \texttt{src/} -- Python scripts for machine learning model development
    \item \texttt{figures/} -- Generated visualizations including Kaplan-Meier curves, ROC plots, and feature importance graphs
    \item \texttt{paper/} -- LaTeX files for manuscript preparation
    \item \texttt{Makefile} -- Automates the process of generating all results and compiling the final manuscript PDF with a single \texttt{make} command
\end{itemize}

\section{Usage}

To reproduce the results:

\begin{verbatim}
git clone https://github.com/CMTs-Draft/DOGMA
cd DOGMA
make
\end{verbatim}

This will preprocess data, train models, generate figures, and compile the LaTeX manuscript.

\section{Acknowledgements}

We acknowledge the support of veterinary oncologists and computational researchers who contributed to the dataset and model validation.
Lee, S., Seung, BJ., Yang, I.S. et al. 1H NMR based urinary metabolites profiling dataset of canine mammary tumors. Sci Data 9, 132 (2022). https://doi.org/10.1038/s41597-022-01229-1
[INSERT ACKNOWLEDGEMENTS] 

\end{document}
