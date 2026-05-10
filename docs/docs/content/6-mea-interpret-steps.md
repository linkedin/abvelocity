# MEA Interpretation and Sizing

**Author**: Reza Hosseini


## Purpose

This document provides step-by-step guidance on how to interpret the MEA report for most standard use cases. We will discuss selecting the best combination and sizing (monetary impact).

### Main Steps (Decide Potential Best Combination/Combinations)

- After running MEA for the primary metrics of interest,
- Check the MEA report. For each combination, you will find the percent delta and delta with p-values.
- The tables are deliberately separated by combination, so the information for each combination is contained within one table.
- Scan the tables and determine which combination is the best.
- You have learned something which is only possible with MEA! You know which combination is the best!
- You may need to make some judgment calls, e.g., if the SWI is negligible for a metric, ignore that metric even if the p-value is significant (such noise is expected in experiment results).

### Next Steps: Sizing (Calculating the Monetary Impact)

- If we are not concerned about reporting individual experiment impacts, we can size using the numbers observed for that combination.
- This is relatively simple, and we convert the metric to dollars as usual.
- SWI for summable metrics, such as signups, is given as a delta sum.
- SWI for ratio metrics, e.g., survival rate, can be obtained by checking the SWI of the numerator and denominator.
- The SWI shown in the table for a metric like survival rate is the SWI for the denominator (which should be sufficient for sizing).
- If you need the change in the denominator, look up that metric's denominator SWI.
- You are done if you only care about the ultimate impact of launching a combination!
- If you want to understand individual impacts and communicate them separately for each experiment, read on.

### Next Steps: Interaction and Relationship to Sizing

This question arises frequently, especially when sizing is involved. Teams want to know if they need to adjust due to MEA.

The answer is nuanced. The interaction of Experiment 1 with Experiment 2 might depend on which variant of, say, Experiment 2 we are considering!
Therefore, any tool that promises a simple Yes/No might be combining results in ways that do not reflect reality if a specific variant of, say, Experiment 2 is launched. This is because that particular variant might have an interaction and change the impact of Experiment 1.

Given the above context, here are the steps:

- Now that you have selected one of a few candidate combinations in the previous steps, you are ready to continue.
- For that given combination, e.g., (Experiment 1:v1,cccccbngnegucndifhdlnivllgvghkfdgcgvjnlbcrdh
 Experiment 2:test), read the SWI (delta sum) for that combination.
- Standardize that SWI into a weekly number.
- Read the individual weekly SWI impacts from the AB platform (ABPlatform) for Experiment 1:v1 and Experiment 2:v2.
- Check if the individual SUM is close to the combination SWI.
- If they are within 10% (rule of thumb), we can size the experiment as usual.
- If there is more than a 10% difference, we should reduce the gains from both experiments.
- See an example of such a case [here](https://docs.google.com/spreadsheets/d/1nDo1eYYLFLcy3L719EJzYhdpz6PPCe28nRg2VV47i24/edit?gid=0#gid=0).

### Interactions in Secondary Metrics in the Context of Sizing

In some cases, you might have primary and secondary metrics.

Primary metrics are those mainly impacted by the experiment and play the major role in sizing.
For example, you may run an experiment that mainly impacts signups but also has a small impact on retention.

- If the impact on secondary metrics is relatively small and more than 80% of the impact is in the primary metrics, you can size the impact of the secondary metrics directly from the individual experiments to save time.
- If not, and secondary metrics, e.g., retention, are very important, you need to repeat the above exercise.
- For example, consider the survival rate. The MEA report includes the SWI on both its numerator and denominator, and you can check if the SUMs are close enough to the combination impact (as both numerator and denominator are indeed summable metrics).