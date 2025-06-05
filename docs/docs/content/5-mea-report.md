# MEA Report

**Author**: Reza Hosseini


### Example MEA report (html / pdf) and how to generate one
- Here is an example of an [HTML](/docs/static//test-results/mea/mea_sim1/mea_test.html)
- The MEA flow will create result tables which could be accessed and read
- It will also generate an html file which can be consumed / hosted in any dashboard.
- You will get the html if you run the MEA flow locally or in a Notebook / Cloud Notebook
- PDF: You can convert the html to pdf if you desire and keep the color coding. Print the pdf after opening it.
- PDF: For printing:
    - Press: "Command + p"
    - After the print UI opens, go to "More settings" and put a tick next to "Keep Background Graphics"
    - This will ensure color coding remains visible.
    - You can now share the PDF instead of html.
- Generate the report within a Python Notebook
    - Capture the resulting html is a variable `html_str`
```python
html_str = """
content of MEA report pasted here
"""
```
    - In a next cell run:
```python
from IPython.core.display import display, HTML
display(HTML(html_str))
```


### Example MEA report using markdown and Docs
- NOTE: Less Readable that HTML since there is no color coding and Docs is not suitable for long tables
- MEA flow also can output a markdown file.
- You can copy the content of that file and right click on a new empty Google Doc, choose: "Paste From Markdown" in Google Docs
- You might need to [activate this feature in Google Docs](https://support.google.com/docs/answer/12014036?hl=en#zippy=%2Cconvert-markdown-to-google-docs-content-on-paste)
- Now you have a google doc with your results! Feel free to remove parts of the content that you do not need.
- The google doc version does not have color coding for significant effects, so in that sense the html report looks nicer.
However you could highlight those in the final table.


### Understanding the MEA report
- See example report: [HTML](/docs/static//test-results/mea/mea_sim1/mea_test.html)
- The report does reiterate some information about the experiments you run analysis for, dates etc:
read those to double check you have entered the correct info.
- Note that this report is based on simulated data therefore the top portion with experiment ids etc are None. But if you generate a report by passing ExptInfo, those will be populated.
- Note that the impacted population in MEA is the union of the impacted population for all experiments involved.
- Next Inspect: Trigger State Count DataFrame: This will tell you how much the group of
experiments overlap. E.g. for two experiments (True, True) is when both experiments triggered.
- Next Inspect: the Interference Rate for Each Experiment, say `Expt i`.
Note that by overlap here we refer to other experiments being triggered when `Expt i` is triggered.
It may or may not be the case that the presence of other treatments change the impact of `Expt i`.
Mathematically, these are conditional distributions of the Trigger State of other experiments for the target population of `Expt i`.
In other words, this tells you how much of the `Expt i` target population
is also targeted by other experiments as well. If our goal is to see if univariate results
of Expt i, hold up a low number here can already indicate that there is a smaller chance it will happen and we can rely on
univariate test results (from AB Platform). We recommend 5% as the threshold based on historical analysis of the data.
Note that this is not guaranteed as it might be the diff is coming from the small overlap, therefore extra investigation is encouraged.
- Also Inspect: Variant Count DataFrame:  To get some idea about different variant combinations and their triggering rates
- There are two remaining tables.
- Table 1: "variant_effect_df_pairs": Skip this one at first. This table could be pretty long because it includes all combinations and metrics. This will be only generated if `end_user_report = True` is passed to the publish method as follows:

```bash
write_path = ""
mea_report = mea.publish(
    write_path=write_path,
    rounding_digits=2,
    html_file_name="mea_test.html",
    end_user_report = True,
)
```

- Table 2: "variant_effect_df_pairs_sig": This is a subset of the above table and only includes the rows with p-values less than 10\%.
We included less than 10\% and not 5\% so that you can see those metrics that could be borderline as well.
- Here is an explanation for the important columns of these two tables (they have the same columns)
    - launch: What happens if this combination is launched compared to control? For scenario based the comparison is with the custom scenario you have passed and that would be clear from the table eg we might want to compare `(enabled, v1)` Launch with `(control, v1)` baseline. This is the impact of launching `expt 1: enabled`, if `expt 2: v1` is already in prod.
    - delta_percent: the percent change
    - ci_percent: the percent change confidence interval
    - delta_sum: This is site-wide-impact for regular metrics or the SWI of numerator for ratio metrics.
    - p-value: p value

