# What is MEA?

**Author**: Reza Hosseini


MEA is an analysis method and tooling to understand the impact of launching multiple features using the data from their corresponding experiments.


### Applications
* **What proportion of the target population of `Expt i` is also targeted by `Expt 2, 3, ...`?** This is a basic use case, in which you can get stats on how much the experiments co-trigger in the same time.
This may already alleviate the concerns. For example if only 5% of the `Expt 1` target population is targeted by `Expt 2`, the individual readouts from Expt 1 are more likely to be reliable.


* **Achieving Accurate Reads When Experiments Interact**: Contrary to popular belief, results from experiment platforms are not necessarily unbiased, even if experiments are run orthogonally. The assumption is that most experiments have minimal interaction, but this can often be false. MEA addresses this issue.

* **Determining the Optimal Combination of Variants Across Multiple Experiments**: For instance, consider two experiments:

  - **Experiment 1**: Changes the text color with variants `control`, `v1`, `v2`, and `v3`.
  - **Experiment 2**: Changes the content with variants `control`, `enabled`, and `test`.

  **Question**: Which combination is optimal to launch? For example, would it be best to launch with `v1` from Experiment 1 and `enabled` from Experiment 2?

* **Scenario-Based Analysis: What Is the Incremental Impact of My Experiment?**: For example, consider the following two experiments:

  - **Experiment 1**: Uses Gen-AI to create card content on the Chooser Page with variants `control`, `v1`, and `v2`.
  - **Experiment 2**: Changes the layout of the Chooser Page with variants `control` and `enabled`.

  **Question**: What would be the incremental impact of Experiment 1 if the `v1` variant is launched?

* **Attribution**: While scenario-based analysis can aid in attribution, it's important to understand its limitations. We can identify the incremental impact of each experiment given that certain variants of other experiments are launched.

For example, suppose `(Experiment 1: v1, Experiment 2: enabled)` is launched, resulting in a 3% increase in subscriptions. Here’s how the impact might break down:

    - The `v1` variant alone contributes 1.5% (compared to production).
    - The `enabled` variant alone contributes 2%.
    - Notice that the combined impact is not simply the sum: 3% ≠ 1.5% + 2% = 3.5%.
    - The total impact might even exceed the sum due to "synergy" (good interaction).

    MEA can help determine incremental impact even when experiments are run concurrently. For instance:
    - It can show the additional gain from Experiment 2’s `enabled` variant as 0.5%.
    - It can also provide the incremental impact of Experiment 1 if Experiment 2 is launched.

* **Is Summable Attribution Possible?** Philosophically, it's not feasible to make attribution fully summable. However, we can use the following approach as an approximation:

    - Calculate individual impacts (by comparing each experiment to its control), resulting in values `e1` and `e2`.
    - Calculate the joint impact, resulting in `e3`.
    - Then, attribute the impacts proportionally:
      - Assign `e3 * (e1 / (e1 + e2))` to Experiment 1.
      - Assign `e3 * (e2 / (e1 + e2))` to Experiment 2.

    This ensures that the impact is summable and attributed based on each experiment's individual effect size.

    **Warning**: This approximation should only be applied when experimental noise is minimal, as all values are estimates. We recommend limiting the number of experiments and variants when precise calculations are essential.


### Requirements

- The experiments are run orthogonally (not hash-aligned). This means the assignments across experiments are independent.
- There are enough samples in the combinations of interest. See the next two sections for more details.
- For applications in tech, there is 2 weeks plus temporal overlap in experiment periods (we want to make sure temporal trends in diffs are averaged out).

### Practical Recommendations for Using MEA

- Although the code does not have limitations on how many experiments to overlap and how many variants each can have,
sample size becomes an issue with two many experiments or variants.
- Suppose there are `k` experiments to overlap each having `m1, m2, ..., mk` variants.
The number that we want to control is `m = m1 * ... * mk`.
For example with two experiments each having 3 variants that number would be `m = 3 * 3 = 9`
and with 3 experiments each having 2 variants, that number would be `m = 2 * 2 * 2`.
- Generally, we recommend keeping `m` smaller than 10. See the next two sections for a more accurate picture.

### Sample Size Guidelines

- This section provides guidelines for understanding sample sizes in Multi-Experiment Analysis (MEA) compared to a simple experiment setup.
- We define "available sample size" as the number of samples that end up in a given "arm", based on population size and triggering rates. In MEA context an arm is essentially a combination of variants to be tested. This is essentially the value which is used to divide the variance by.
- By determining the "available sample size" with MEA, you can apply this to estimate the feasibility of running your experiment. By (1) identify the sample size you need for your metric; (2) compare the needed sample size with the "available sample size" calculated here.

- **Simple Experiment Case**
    - Assume `N` is the number of units in the population.
    - Let `r` (a value between 0 and 100) represent the triggering rate.
    - Assume there are `m1` equally weighted variants.
    - Then the sample size for each variant can be calculated as:
      ```
      Sample Size = `N * r / m1`
      ```
      This formula represents the number of units per arm for hypothesis testing.

- **Multi-Experiment Analysis (MEA) Case**
    - Recall that `m = m1 * ... * mk` represents the total number of possible combinations of experiments.
    - **Same Subset Triggering Case (Full Overlap)**
      - In the simplest (and worst-case) scenario, all experiments trigger on the same subset, with a triggering rate of `r`.
      - Here, the sample size is:
        ```
        Sample Size = N * r / m = N * r / (m1 * ... * mk)
        ```

    - **Two Experiments Case**
      - Let’s consider two experiments, Expt 1 and Expt 2.
      - Define `r(1, 1)` as the co-triggering rate, where `1` in position `i` indicates Expt `i` is triggered.
      - In general `r(., .)` determines triggering with `1` in each `i-th` position indicating a trigger.
      For example `r(1, 0)` is the rate at which experiment 1 triggers and second experiment does not.
      - The sample size is calculated as:
        ```
        Sample Size = (N * r(1, 1) / m)
                    + (N * r(0, 1) / m2)
                    + (N * r(1, 0) / m1)
        ```

    - **Three Experiments Case**
      - For three experiments, the co-triggering rate `r(., ., .)` determines triggering with `1` in each `i-th` position indicating a trigger.
      - The sample size formula for this case is:
        ```
        Sample Size = (N * r(1, 1, 1) / m)
                    + (N * r(0, 1, 1) / (m2 * m3))
                    + (N * r(1, 0, 1) / (m1 * m3))
                    + (N * r(1, 1, 0) / (m1 * m2))
                    + (N * r(0, 0, 1) / m3)
                    + (N * r(0, 1, 0) / m2)
                    + (N * r(1, 0, 0) / m1)
        ```
- The above formulas do extend to overlapping experiments more than three.
- The intuition behind the above formulas is as follows. Depending on the partition
we consider, the number of samples which will be available in the estimation depends
on the number of variants (`m_i`) of the experiments triggered within that partition.

### ✅ How to perform MEA (Logical Steps)

- Choose a few (1 to 3) key metrics upfront as primary metrics to focus on.
This practice is the recommended approach for AB testing in general and not only restricted to MEA.
- If you are choosing a metric on a particular dimension, it is recommended to also look into the overall
- The metrics you are choosing obviously should depend on the change you are making and your target population.
- Perform MEA

### 🛑 What not to do\!

- [P-value value hacking](https://en.wikipedia.org/wiki/Data\_dredging) by running MEA on different combinations and metrics until you see something positive about your change. With the new power, it comes new responsibility\! Do not abuse MEA\!
