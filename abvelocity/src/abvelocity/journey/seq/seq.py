# BSD 2-CLAUSE LICENSE

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# #ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LAW, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

import os
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from abvelocity.journey.seq.conversion_query import ConversionQuery
from abvelocity.journey.seq.seq_info import (
    CONSECUTIVE_DEDUPED,
    FULLY_DEDUPED,
    MAP,
    SEQ_INFO_LIST,
    UNDEDUPED,
    SeqInfo,
)
from abvelocity.journey.seq.seq_query import SeqQuery
from abvelocity.journey.viz.journey_barchart import JourneyBarchart
from abvelocity.journey.viz.sankey_plot import SankeyPlot
from abvelocity.journey.viz.sunburst_plot import SunburstPlot
from abvelocity.param.io_param import IOParam
from abvelocity.param.join_query import JoinQuery

# Default time column
TIME_COL = "time"

# Default event column
EVENT_COL = "event"

# Sequence column
SEQ_COL = "event_seq"


PLOT_VALUE_COLS = ["percent", "count"]


@dataclass
class Seq:
    """Dataclass to hold journey parameters."""

    event_table_name: str = ""
    """Event table name or a query to extract events.
    This query / table result includes a data schema of the form:
        `UNIT_COL`, `TIME_COL`, `EVENT_COL`, `self.partition_by_cols`

        - `UNIT_COL` (str) is typically the main unit of interest eg member for member data
        - `TIME_COl` (str / int) is a measure of time and if time is to be used to order later on, we will rely on its order
        - `EVENT_COL` (str) includes the events of interest
        - `self.partition_by_cols` This is a collection of columns to partition by in order to define journey boundaries, for example for usage
        data this could be a session id, or some kind of reference id.

    """
    event_queries_dict: Dict[str, str] = field(default_factory=dict)
    """Event queries. This is useful if multiple separate queries should be executed to construct a final event table.
    The choice to allow for multiple queries is as follows:
    These queries are to be executed by a platform and sometimes users need to give us a few queries to be run to make their
    set up simpler or due to computational issues (can materialize several tables in the middle).
    """
    create_table_prefix: str = ""
    """A prefix table name to be used in creating / storing tables."""
    partition_by_cols: List[str] = field(default_factory=list)
    """Columns to partition by and determine journey / seq boundaries."""
    start_date: str = ""
    """Start date used in event queries"""
    end_date: str = ""
    """End date used in event queries."""
    conditions: List[str] = field(default_factory=list)
    """An optional list of conditions to be applied to event queries."""
    seq_info_list: List[SeqInfo] = field(default_factory=lambda: SEQ_INFO_LIST)
    """The sequence types we want to generate with their output table name."""
    seq_queries_dict: Dict[str, str] = field(default_factory=dict)
    """Queries to generate seq data."""
    join_queries_dict: Dict[str, str] = field(default_factory=dict)
    """Queries to perform joins after initial seq tables are generated."""
    order_list: Optional[List[str]] = None
    """A custom order to use instead of time, time is still used when there are ties or label does not appear in `order_list`."""
    post_joins: Optional[List[JoinQuery]] = None
    """A list of joins given in the `Join` dataclass format.
    These joins are meant to update these tables with their names stored in these attributes:

        - "consecutive_output_table_name",
        - "fully_output_table_name",
        - "undeduped_output_table_name",
        - "map_output_table_name"

    Here is how the joins work

        - The joins occur after the sequence tables are generated.
        - The joins includ the `right_table` only and as the `left_table`, we use each of the generated seq tables.
        - The joins are incremental. This means we do one join at a time and we use the resulting table as the `left_table`
            for the next join.
        - The intermediate joined tables have the same name as original table plus f"_join{i}" depending on the order.
        - The final joined table name is the same as the original table plus "_joined"
        - We update the name in the given attributes with the "_joined" version.
    """

    event_color_dict: Optional[Dict[str, str]] = None
    io_param: Optional[IOParam] = None

    @abstractmethod
    def gen_event_queries(self, **kwargs) -> Dict[str, str]:
        """
        Abstract method to generate event-related queries and stores them in
        `self.event_queries_dict`.
        Note that this may not be needed if `self.event_table_name` is a prepared table.
        This will update the final event_table_name as well if needed.
        """
        pass

    def gen_seq_queries(self) -> Dict[str, str]:
        """
        Generates sequence-related queries and stores them in `self.seq_queries_dict`.
        There are two queries stored in the dict:

            - "consecutive_deduped_seq": This version will convert a journey of the form a, a, b, b, c, a -> [a, b, c, a]
                Which means it will eliminate consecutive occurrences of events.
            - "fully_deduped_seq": This version will convert the above journey to: [a, b, c]
                Which means it will only keep the first occurrence of each event.

        The expected output schema for both of these tables is as follows:

            `UNIT_COL`, "event_seq", `self.partition_by_cols`

        where:

            - `UNIT_COL` (str) is typically the main unit of interest eg member for member data
            -  "event_seq" (str) includes the journey / seq in array format eg [a, b, c]
            - `self.partition_by_cols` This is a collection of columns to partition by in order to define journey boundaries, for example for usage
            data this could be a session id, or some kind of reference id.

        """

        for seq_info in self.seq_info_list:
            if not seq_info.output_table_name:
                seq_info.output_table_name = (
                    f"{self.create_table_prefix}_{seq_info.deduping_method}_seq"
                )

        for seq_info in self.seq_info_list:
            deduping_method = seq_info.deduping_method
            self.seq_queries_dict[deduping_method] = SeqQuery(
                event_table_name=self.event_table_name,
                time_col=TIME_COL,
                event_col=EVENT_COL,
                output_table_name=seq_info.output_table_name,
                partition_by_cols=self.partition_by_cols,
                deduping_method=deduping_method,
                max_seq_index=seq_info.max_seq_index,
            ).gen()
            print(f"\n\n\n*** {seq_info.deduping_method}: {self.seq_queries_dict[deduping_method]}")

        return self.seq_queries_dict

    def gen_join_queries(self):
        """Generates join queries."""
        if not self.post_joins:
            return

        # Iterate through the collected table names and produce the join queries
        for seq_info in self.seq_info_list:
            table_name = seq_info.output_table_name
            # At first we use the original table_name as the `left_table`
            table_name_for_join = seq_info.output_table_name
            for i, join in enumerate(self.post_joins):
                # We assign the joined table so far to the `left_table`
                # Note that the `right_table` is already populated
                join.left_table = table_name_for_join
                # Determine the output table name for the current join
                # For the last table we use a simple name without index
                output_table_name = (
                    f"{table_name}_join{i}"
                    if i < len(self.post_joins) - 1
                    else f"{table_name}_joined"
                )
                join.output_table_name = output_table_name
                # Generate and store the SQL query
                query = join.gen()
                self.join_queries_dict[join.output_table_name] = query
                print(f"\n\n\n*** query: {table_name}:\n {query}")
                # The next join will use the newly joined table as `left_table`
                # This is because the join is incremental
                table_name_for_join = output_table_name

        # Reset (append suffix to) the original output table names
        for seq_info in self.seq_info_list:
            current_value = seq_info.output_table_name
            if current_value:
                seq_info.output_table_name = f"{current_value}_joined"

    def gen_sunburst_plots(
        self,
        conditions: List[str] = ["TRUE = TRUE"],
        count_distinct_col: str = None,
        value_cols: List[str] = PLOT_VALUE_COLS,
        deduping_methods: List[str] = [UNDEDUPED, CONSECUTIVE_DEDUPED, FULLY_DEDUPED],
    ):
        """Generates and saves sunburst plots for the sequence tables."""

        if self.io_param is None:
            raise ValueError("io_param must be set to use this method.")

        save_path0 = os.path.join(self.io_param.save_path, "sunburst")
        os.makedirs(save_path0, exist_ok=True)

        sunburst_io_param = IOParam(
            cursor=self.io_param.cursor,
            print_to_html=self.io_param.print_to_html,
            save_path=save_path0,
            file_name_suffix=self.io_param.file_name_suffix,
        )

        sunburst_plot = SunburstPlot(io_param=sunburst_io_param)

        res = {}
        for seq_info in self.seq_info_list:
            deduping_method = seq_info.deduping_method
            table_name = seq_info.output_table_name
            max_seq_index = seq_info.max_seq_index
            if deduping_method in deduping_methods and table_name:
                for value_col in value_cols:
                    print(
                        f"\n***: Generating sunbursts for table: {table_name}: conditions: {conditions}, quantity: {value_col}"
                    )
                    res[(deduping_method, value_col)] = sunburst_plot.gen(
                        table_name=table_name,
                        value_col=value_col,
                        conditions=conditions,
                        max_seq_index=max_seq_index,
                        count_distinct_col=count_distinct_col,
                        color_dict=self.event_color_dict,
                    )
        return res

    def gen_sankey_plots(
        self,
        conditions: List[str] = ["TRUE = TRUE"],
        count_distinct_col: str = None,
        add_end_state: bool = False,
        distinct_nodes_by_stage: bool = False,
        value_cols: List[str] = PLOT_VALUE_COLS,
        deduping_methods: List[str] = [UNDEDUPED, CONSECUTIVE_DEDUPED, FULLY_DEDUPED],
    ):
        """Generates and saves sankey plots for the sequence tables."""

        if self.io_param is None:
            raise ValueError("io_param must be set to use this method.")

        save_path0 = os.path.join(self.io_param.save_path, "sankey")
        os.makedirs(save_path0, exist_ok=True)

        sankey_io_param = IOParam(
            cursor=self.io_param.cursor,
            print_to_html=self.io_param.print_to_html,
            save_path=save_path0,
            file_name_suffix=self.io_param.file_name_suffix,
        )

        sankey_plot = SankeyPlot(io_param=sankey_io_param)

        res = {}
        for seq_info in self.seq_info_list:
            deduping_method = seq_info.deduping_method
            table_name = seq_info.output_table_name
            max_seq_index = seq_info.max_seq_index
            if deduping_method in deduping_methods and table_name:
                for value_col in value_cols:
                    print(
                        f"\n***: Generating sunbursts for table: {table_name}: conditions: {conditions}, quantity: {value_col}"
                    )
                    res[(table_name, value_col)] = sankey_plot.gen(
                        table_name=table_name,
                        value_col=value_col,
                        conditions=conditions,
                        max_seq_index=max_seq_index,
                        count_distinct_col=count_distinct_col,
                        color_dict=self.event_color_dict,
                        add_end_state=add_end_state,
                        distinct_nodes_by_stage=distinct_nodes_by_stage,
                    )
        return res

    def gen_journey_barcharts(
        self,
        conditions: List[str] = ["TRUE = TRUE"],
        count_distinct_col: str = None,
        sort_event_array: bool = False,
        deduping_methods: List[str] = [UNDEDUPED, CONSECUTIVE_DEDUPED, FULLY_DEDUPED, MAP],
    ):
        """Generates and saves journey barcharts for the sequence tables."""

        if self.io_param is None:
            raise ValueError("io_param must be set to use this method.")

        save_path0 = os.path.join(self.io_param.save_path, "barchart")
        os.makedirs(save_path0, exist_ok=True)

        barchart_io_param = IOParam(
            cursor=self.io_param.cursor,
            print_to_html=self.io_param.print_to_html,
            save_path=save_path0,
            file_name_suffix=self.io_param.file_name_suffix,
        )

        gen_barchart = JourneyBarchart(io_param=barchart_io_param)

        res = {}
        for seq_info in self.seq_info_list:
            deduping_method = seq_info.deduping_method
            table_name = seq_info.output_table_name

            print(f"\n***: Generating barcharts for table: {table_name}: conditions: {conditions}")
            is_map = deduping_method == MAP
            res[table_name] = gen_barchart.gen(
                table_name=table_name,
                conditions=conditions,
                count_distinct_col=count_distinct_col,
                sort_event_array=sort_event_array,
                is_map=is_map,
            )

        return res

    def get_seq_info(self, deduping_method: Optional[str]) -> SeqInfo:
        """Given a `deduping_method`, this returns seq_info with that deduping method.
        Args
        deduping_method (str): The `deduping_method` for calculating conversions.
            If not passed, we use `FULLY_DEDUPED`

        Returns:
            seq_info (SeqInfo): The corresponding `SeqInfo` instance.

        """
        if not deduping_method:
            deduping_method = FULLY_DEDUPED

        return next(
            (
                seq_info
                for seq_info in self.seq_info_list
                if seq_info.deduping_method == deduping_method
            ),
            None,  # Default to None if the method is not found
        )

    def gen_seq_summary_query(
        self,
        deduping_method: Optional[str] = None,
        count_distinct_col: Optional[str] = None,
        group_by_cols: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,  # New argument for WHERE clauses
        time_unit: str = "minutes",
        time_col_format: str = "unix_ms",
    ) -> str:
        """
        Generates a SQL query to summarize event sequences.

        Args:
            deduping_method (str, optional): The deduping_method for calculating conversions.
                If not passed, we use the preferred method via `get_seq_info`.
            count_distinct_col (str, optional): If provided, counts distinct values of this column
                instead of total event_seq count. Defaults to None.
            group_by_cols (list of str, optional): A list of column names to group the results by.
                Defaults to None.
            conditions (List[str], optional): A list of SQL WHERE clause conditions (e.g., ["a = 'b'", "x = 1"]).
                These will be combined with AND. Defaults to None.
            time_unit (str, optional): The unit for the average time duration.
                Can be 'seconds', 'minutes', 'hours', or 'days'. Defaults to 'minutes'.
            time_col_format (str, optional): Specifies the format of `seq_start_time` and `seq_end_time`.
                Valid options:
                - 'unix_ms': Unix timestamp in milliseconds (e.g., 1678886400000).
                             Will use FROM_UNIXTIME(timestamp_col / 1000.0).
                - 'unix_s': Unix timestamp in seconds (e.g., 1678886400).
                            Will use FROM_UNIXTIME().
                - 'string': String formatted timestamp (e.g., '2023-03-15 10:00:00').
                            Will use CAST(... AS TIMESTAMP).
                - 'timestamp': Already a TIMESTAMP type. No casting applied.
                Defaults to 'unix_ms'.

        Returns:
            str: The generated SQL query.
        """

        # Map common time units to the expected SQL literal for DATE_DIFF
        time_unit_mapping = {
            "second": "SECOND",
            "seconds": "SECOND",
            "minute": "MINUTE",
            "minutes": "MINUTE",
            "hour": "HOUR",
            "hours": "HOUR",
            "day": "DAY",
            "days": "DAY",
        }

        sql_time_unit = time_unit_mapping.get(time_unit.lower())
        if not sql_time_unit:
            print(
                f"Warning: Invalid or unhandled time_unit '{time_unit}'. Falling back to 'MINUTE'."
            )
            sql_time_unit = "MINUTE"

        # Validate time_col_format
        allowed_time_formats = {"unix_ms", "unix_s", "string", "timestamp"}
        if time_col_format not in allowed_time_formats:
            print(
                f"Warning: Invalid time_col_format '{time_col_format}'. Falling back to 'unix_ms'."
            )
            time_col_format = "unix_ms"

        # Use get_seq_info to determine the table name based on fallback logic
        seq_info = self.get_seq_info(deduping_method)
        if not seq_info:
            raise ValueError(
                f"Could not find a sequence info for deduping method '{deduping_method}' or its fallbacks."
            )
        table_name = seq_info.output_table_name

        select_parts = []
        group_by_parts = []

        # 1. Handle Group By Columns
        if group_by_cols:
            select_parts.extend(group_by_cols)
            group_by_parts.extend(group_by_cols)

        # 2. Count distinct members or events
        if count_distinct_col:
            select_parts.append(f"COUNT(DISTINCT {count_distinct_col}) AS seq_count")

        # 3. Calculate average seq_length (total events in a journey)
        select_parts.append("AVG(seq_length) AS avg_seq_length")

        # 4. Prepare time columns based on time_col_format
        start_time_col_expr = "seq_start_time"
        end_time_col_expr = "seq_end_time"

        if time_col_format == "unix_ms":
            start_time_col_expr = "FROM_UNIXTIME(seq_start_time / 1000.0)"
            end_time_col_expr = "FROM_UNIXTIME(seq_end_time / 1000.0)"
        elif time_col_format == "unix_s":
            start_time_col_expr = "FROM_UNIXTIME(seq_start_time)"
            end_time_col_expr = "FROM_UNIXTIME(seq_end_time)"
        elif time_col_format == "string":
            start_time_col_expr = "CAST(seq_start_time AS TIMESTAMP)"
            end_time_col_expr = "CAST(seq_end_time AS TIMESTAMP)"
        # If time_col_format is 'timestamp', no casting is needed, so expressions remain as is.

        # 5. Calculate average duration using the prepared time column expressions
        select_parts.append(
            f"AVG(DATE_DIFF('{sql_time_unit}', {start_time_col_expr}, {end_time_col_expr})) AS average_time_{time_unit}"
        )

        query = f"SELECT {', '.join(select_parts)}\nFROM {table_name}"

        # 6. Add WHERE clause if conditions are provided
        if conditions:
            # Combine all conditions with ' AND '
            query += f"\nWHERE {' AND '.join(conditions)}"

        if group_by_parts:
            query += f"\nGROUP BY {', '.join(group_by_parts)}"

        return query

    def get_seq_summary(
        self,
        deduping_method: Optional[str] = None,
        count_distinct_col: Optional[str] = None,
        group_by_cols: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,  # Pass conditions to gen_seq_summary_query
        time_unit: str = "minutes",
        time_col_format: str = "unix_ms",
    ):
        """
        Generates a dataframe which summarize event sequences.

        Args:
            deduping_method (str, optional): The deduping_method for calculating conversions.
                If not passed, we use `FULLY_DEDUPED`.
            count_distinct_col (str, optional): If provided, counts distinct values of this column
                instead of total event_seq count. Defaults to None.
            group_by_cols (list of str, optional): A list of column names to group the results by.
                Defaults to None.
            conditions (List[str], optional): A list of SQL WHERE clause conditions (e.g., ["a = 'b'", "x = 1"]).
                These will be combined with AND. Defaults to None.
            time_unit (str, optional): The unit for the average time duration.
                Can be 'seconds', 'minutes', 'hours', or 'days'. Defaults to 'minutes'.
            time_col_format (str, optional): Specifies the format of `seq_start_time` and `seq_end_time`.
                Valid options:
                - 'unix_ms': Unix timestamp in milliseconds (e.g., 1678886400000).
                    Will use FROM_UNIXTIME(timestamp_col / 1000.0).
                - 'unix_s': Unix timestamp in seconds (e.g., 1678886400).
                    Will use FROM_UNIXTIME().
                - 'string': String formatted timestamp (e.g., '2023-03-15 10:00:00').
                    Will use CAST(... AS TIMESTAMP).
                - 'timestamp': Already a TIMESTAMP type. No casting applied.
                Defaults to 'unix_ms'.

        Returns:
            pd.DataFrame: The generated dataframe.
        """

        query = self.gen_seq_summary_query(
            deduping_method=deduping_method,
            count_distinct_col=count_distinct_col,
            group_by_cols=group_by_cols,
            conditions=conditions,  # Pass the new conditions argument
            time_unit=time_unit,
            time_col_format=time_col_format,  # Ensure time_col_format is passed
        )

        if self.io_param is None:
            raise ValueError("io_param must be set to use this method.")

        # This part assumes self.io_param.cursor.get_df exists and works.
        # You might need to import pandas as pd if not already done in your actual file.
        # import pandas as pd # Example import
        return self.io_param.cursor.get_df(query=query).df

    def calc_conversion(
        self,
        numerator_list: List[str],
        denominator_list: List[str],
        count_distinct_col: Optional[str] = None,
        conditions: Optional[List[str]] = field(default_factory=list),
        group_by_cols: Optional[List[str]] = field(default_factory=list),
        deduping_method: Optional[str] = None,
    ):
        """
        Calculates conversion rates based on specified event sequences and returns
        the generated SQL query along with the resulting DataFrame.

        This method leverages the `ConversionQuery` class to construct a SQL query.
        The conversion rate is determined by the ratio of entities (or rows) that
        satisfy the numerator criteria versus those that satisfy the denominator criteria,
        based on the content of an array column (likely event sequences or tags).

        Args:
            numerator_list (List[str]): A list of event tags or values that define the
                numerator for the conversion calculation. Entities (rows or distinct items)
                that have any of these tags will be counted.
            denominator_list (List[str]): A list of event tags or values that define the
                denominator for the conversion calculation. Entities (rows or distinct items)
                that have any of these tags will be counted.
            count_distinct_col (Optional[str], optional): The column to use for
                `COUNT(DISTINCT)`. If provided (e.g., 'user_id' or 'session_id'), the
                conversion will be based on unique values in this column. If `None`, it
                defaults to counting all relevant rows. Defaults to `None`.
            conditions (Optional[List[str]], optional): A list of additional SQL WHERE
                conditions (e.g., "event_date >= '2023-01-01'"). These conditions will
                be combined with ' AND ' in the generated query. Defaults to an empty list.
            group_by_cols (Optional[List[str]], optional): A list of columns to group the
                conversion results by (e.g., 'country', 'DATE(event_timestamp)'). If
                provided, the query will return conversion rates per group. Defaults to
                an empty list.
            deduping_method: The `deduping_method` for calculating conversions.
                If not passed, we use `FULLY_DEDUPED`

        Returns:
            Dict[str, Union[str, pd.DataFrame]]: A dictionary containing:
                - **'query' (str)**: The generated Presto SQL query string.
                - **'df' (pd.DataFrame)**: A Pandas DataFrame containing the results of
                  executing the query. This DataFrame will be empty if `self.io_param`
                  is `None`.

        Raises:
            ValueError: If `numerator_list` or `denominator_list` are empty.
                (Validation handled by `ConversionQuery`).
            TypeError: If `count_distinct_col` is not a string, or if `conditions`
                or `group_by_cols` are not lists of strings.
                (Validation handled by `ConversionQuery`).
        """

        table_name = self.get_seq_info(deduping_method).output_table_name

        query = ConversionQuery(
            table_name=table_name,
            array_col=SEQ_COL,
            numerator_list=numerator_list,
            denominator_list=denominator_list,
            count_distinct_col=count_distinct_col,
            conditions=conditions,
            group_by_cols=group_by_cols,
        ).gen()

        df = None
        if self.io_param is not None:
            df = self.io_param.cursor.get_df(query).df

        return {"query": query, "df": df}

    def exec_event_queries(self):
        """Executes all event queries."""

        if self.io_param is None:
            raise ValueError("io_param must be set to use this method.")
        if self.event_queries_dict:
            for k, query in self.event_queries_dict.items():
                print(f"\n*** event query: {k} will be executed")
                self.io_param.cursor.execute_multi(query=query)

    def exec_seq_queries(self):
        """Executes all sequence queries."""

        if self.io_param is None:
            raise ValueError("io_param must be set to use this method.")

        if self.seq_queries_dict:
            for k, query in self.seq_queries_dict.items():
                print(f"\n*** seq query: {k} will be executed")
                self.io_param.cursor.execute_multi(query=query)

    def exec_join_queries(self):
        """Executes all post join queries."""

        if self.io_param is None:
            raise ValueError("io_param must be set to use this method.")

        if self.join_queries_dict:
            for k, query in self.join_queries_dict.items():
                print(f"\n*** seq query: {k} will be executed")
                self.io_param.cursor.execute_multi(query=query)
