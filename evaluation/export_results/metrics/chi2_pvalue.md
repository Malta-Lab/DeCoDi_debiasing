| Mode          | Type      | Classification | Evaluator 1 | Evaluator 2 | Chi2 Statistic | P-Value       |
|---------------|-----------|----------------|-------------|-------------|----------------|---------------|
| Nurse         | Original  | Gender         | Evaluator 1       | Evaluator 2       | 0.50           | 0.478         |
| Nurse         | Debiased  | Gender         | Evaluator 1       | Evaluator 2       | 2.67           | 0.102         |
| Nurse         | Original  | Gender         | Evaluator 1       | GPT         | 0.00           | 1.000         |
| Nurse         | Debiased  | Gender         | Evaluator 1       | GPT         | 0.00           | 1.000         |
| Nurse         | Original  | Gender         | Evaluator 2       | GPT         | 0.50           | 0.481         |
| Nurse         | Debiased  | Gender         | Evaluator 2       | GPT         | 3.17           | 0.075         |
| Firefighter   | Original  | Race           | Evaluator 1       | Evaluator 2       | 1.36           | 0.507         |
| Firefighter   | Debiased  | Race           | Evaluator 1       | Evaluator 2       | 8.31           | 0.040         |
| Firefighter   | Original  | Race           | Evaluator 1       | GPT         | 3.69           | 0.297         |
| Firefighter   | Debiased  | Race           | Evaluator 1       | GPT         | 61.90          | 2.31e-13      |
| Firefighter   | Original  | Race           | Evaluator 2       | GPT         | 1.46           | 0.692         |
| Firefighter   | Debiased  | Race           | Evaluator 2       | GPT         | 62.95          | 1.38e-13      |
| CEO           | Original  | Apparent Age   | Evaluator 1       | Evaluator 2       | 96.40          | 1.17e-21      |
| CEO           | Debiased  | Apparent Age   | Evaluator 1       | Evaluator 2       | 21.32          | 2.34e-05      |
| CEO           | Original  | Apparent Age   | Evaluator 1       | GPT         | 227.62         | 3.74e-50      |
| CEO           | Debiased  | Apparent Age   | Evaluator 1       | GPT         | 26.45          | 1.80e-06      |
| CEO           | Original  | Apparent Age   | Evaluator 2       | GPT         | 50.72          | 9.67e-12      |
| CEO           | Debiased  | Apparent Age   | Evaluator 2       | GPT         | 1.42           | 0.491         |



\begin{table}[]
\begin{tabular}{lllllll}
Mode        & Type     & Classification & Evaluator 1 & Evaluator 2 & Chi2 Statistic & P-Value  \\
Nurse       & Original & Gender         & Evaluator 1       & Evaluator 2       & 0.50           & 0.478    \\
Nurse       & Debiased & Gender         & Evaluator 1       & Evaluator 2       & 2.67           & 0.102    \\
Nurse       & Original & Gender         & Evaluator 1       & GPT         & 0.00           & 1.000    \\
Nurse       & Debiased & Gender         & Evaluator 1       & GPT         & 0.00           & 1.000    \\
Nurse       & Original & Gender         & Evaluator 2       & GPT         & 0.50           & 0.481    \\
Nurse       & Debiased & Gender         & Evaluator 2       & GPT         & 3.17           & 0.075    \\
Firefighter & Original & Race           & Evaluator 1       & Evaluator 2       & 1.36           & 0.507    \\
Firefighter & Debiased & Race           & Evaluator 1       & Evaluator 2       & 8.31           & 0.040    \\
Firefighter & Original & Race           & Evaluator 1       & GPT         & 3.69           & 0.297    \\
Firefighter & Debiased & Race           & Evaluator 1       & GPT         & 61.90          & 2.31e-13 \\
Firefighter & Original & Race           & Evaluator 2       & GPT         & 1.46           & 0.692    \\
Firefighter & Debiased & Race           & Evaluator 2       & GPT         & 62.95          & 1.38e-13 \\
CEO         & Original & Apparent Age   & Evaluator 1       & Evaluator 2       & 96.40          & 1.17e-21 \\
CEO         & Debiased & Apparent Age   & Evaluator 1       & Evaluator 2       & 21.32          & 2.34e-05 \\
CEO         & Original & Apparent Age   & Evaluator 1       & GPT         & 227.62         & 3.74e-50 \\
CEO         & Debiased & Apparent Age   & Evaluator 1       & GPT         & 26.45          & 1.80e-06 \\
CEO         & Original & Apparent Age   & Evaluator 2       & GPT         & 50.72          & 9.67e-12 \\
CEO         & Debiased & Apparent Age   & Evaluator 2       & GPT         & 1.42           & 0.491   
\end{tabular}
\end{table}