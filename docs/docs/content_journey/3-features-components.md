---
title: Features and Components
id: features-components
---

**Author**: Reza Hosseini


## Technical Components

The framework consists of core components, detailed below with their supported functionalities.

### Data Ingestion
Collects event data (e.g., event type, timestamp, user ID) from platform interactions, including A/B test assignments for experimentation analysis. Supports configuration-driven ingestion to define event schemas and enrichment logic.

### Sequence Generation
Processes raw events into sequences, supporting multiple types:

| Sequence Type         | Definition                                      | Example                | Use Case                                                                 |
|-----------------------|------------------------------------------------|------------------------|--------------------------------------------------------------------------|
| Undeduped             | All events in order, no deduplication          | `[a, a, b, b, a]`      | Understand repeated actions in user journeys                              |
| Consecutive Deduped   | Removes consecutive duplicates                 | `[a, b, a]`            | Analyze journeys through events, track returning to the same event        |
| Fully Deduped         | Keeps first occurrence of each event           | `[a, b]`               | Understand event order, ignoring repeated events                          |
| Count Map             | Counts event occurrences                       | `{a:3, b:2}`           | Compact representation of events and their frequency                      |
| Basket/Set            | Tracks unique events, ignoring order           | `{a, b}`               | Focus on events present in the sequence, not their order                  |

### Ordering Logic
Supports methods to order events within sequences, addressing timestamp inaccuracies:

| Order Type            | Definition                                      | Use Case                                                                 |
|-----------------------|------------------------------------------------|--------------------------------------------------------------------------|
| Timestamp-Based       | Orders events by recorded timestamp            | Natural ordering for time-based analysis                                 |
| Custom Order          | Uses user-defined logic (e.g., reference IDs)  | Handles imprecise timestamps or domain-specific event flows              |


### Metrics Calculation
Computes quantitative metrics to analyze journeys, supporting complex conditional logic:

| Metric                | Definition                                      | Use Case                                                                 |
|-----------------------|------------------------------------------------|--------------------------------------------------------------------------|
| Conversion Probability | Probability of event B given event A (or sets) | Measure transition likelihood (e.g., from form view to submission)       |
| Sequence Count        | Number of occurrences of a sequence            | Quantify frequency of specific user paths                                |
| Sequence Percentage   | Percentage of sequences matching a pattern     | Compare prevalence of journeys across dimensions (e.g., device type)     |
| Sequence Length       | Number of events in a sequence                 | Identify overly complex flows requiring simplification                   |
| Time Taken            | Time delta between start and end               | Detect delays in user flows (e.g., slow checkout processes)              |

**Note**: Metrics support distinct counts on identifiers (e.g., user ID, session ID) and allow events A/B to be sets for flexible analysis.

### Visualization
Generates interactive visualizations to explore journeys, with slicing capabilities:

| Visualization Type    | Description                                     | Use Case                                                                 |
|-----------------------|------------------------------------------------|--------------------------------------------------------------------------|
| Sankey Plot           | Visualizes flow transitions between events      | Identify drop-offs or common paths in user flows (e.g., checkout funnel) |
| Sunburst Plot         | Displays hierarchical sequence patterns         | Explore nested journey structures by dimensions (e.g., region, device)   |
| Timeseries Plot       | Tracks journey frequency over time             | Monitor trends or detect anomalies in user behavior                      |
| Bar Chart             | Compares event or sequence counts               | Analyze distribution of events across categories (e.g., platforms)       |

**Note**: Sunburst plots offer enhanced interactivity over Sankey plots, supporting drill-down by dimensions.

### Data Processing
Uses scalable pipelines, such as Near Real Time (NRT) query engines like Trino for live analysis and batch processing with tools like Spark for aggregated metrics, to handle large-scale event data. Configuration-driven pipelines generate queries for both NRT (Trino) and batch (Spark) modes, supporting seamless integration with orchestration tools.

### Pipeline Types
Details the processing modes for journey analysis:

| Pipeline Type         | Description                                     | Tools/Example            | Use Case                                                                 |
|-----------------------|------------------------------------------------|--------------------------|--------------------------------------------------------------------------|
| Near Real Time (NRT)  | Processes events with low latency for live queries | Trino                    | Interactive slicing, ad-hoc analysis (e.g., war-room debugging)          |
| Batch Processing      | Aggregates metrics and generates offline tables | Spark                    | Daily anomaly detection, visualization generation, reporting              |

### Slicing
Enables analysis by dimensions (e.g., device type, region) with configuration-driven joins to event tables or external data sources, minimizing setup effort for custom slices.

## Non-Functional Requirements

The framework is designed to meet stringent non-functional requirements, ensuring usability, scalability, and performance for large-scale journey analysis. The following table summarizes these requirements:

| Requirement            | Criteria                                          | Description                                                                 |
|-----------------------|--------------------------------------------------|-----------------------------------------------------------------------------|
| Easy Onboarding       | Configuration in less than 1 hour                        | Users can define events, joins, and parameters quickly, with abstracted sequence generation logic. Flat configs simplify setup. |
| Easy Event Updates    | Add/change events in less than 10 minutes                | Adding or modifying events requires minimal effort via straightforward config updates. |
| Easy Scheduling       | Seamless flow scheduling                        | Generated queries integrate with Trino (NRT) or Spark (batch), enabling effortless scheduling with orchestration tools. |
| Scalability           | Handle billions of event rows                   | Supports large-scale operations (e.g., union of billion-row tables) with robust query generation independent of scheduling. |
| NRT Latency           | Interactive updates in less than 30 seconds              | Trino-powered NRT mode ensures low-latency slicing and ad-hoc analysis for interactive use cases. |

