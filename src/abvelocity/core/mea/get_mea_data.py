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
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini

import gc
import resource
from typing import Optional

from abvelocity.core.get_data.data_container import DataContainer
from abvelocity.core.get_data.db_interact import CreateDBTableSpecs
from abvelocity.core.get_data.get_multi_expt_data import get_multi_expt_data
from abvelocity.core.get_data.get_u_metrics_query_from_info import get_u_metrics_query_from_info
from abvelocity.core.get_data.join_expt_with_metric_df import join_expt_with_metric_df
from abvelocity.core.get_data.join_expt_with_metric_queries import join_expt_with_metric_queries
from abvelocity.core.param.analysis_info import AnalysisInfo
from abvelocity.core.param.constants import VARIANT_COL
from abvelocity.core.param.expt_info import MultiExptInfo
from abvelocity.core.param.io_param import IOParam
from abvelocity.core.param.metric import get_u_metrics

EXPT_METRIC_DF_MATERIALIZE_TYPES = ["Pandas", "Query", "DBTable"]

EXPT_METRIC_DF_MATERIALIZE_SPECS = [
    None,
    ("Pandas",),
    ("DBTable",),
    ("DBTable", "Pandas"),
]


def get_mea_data(
    io_param: IOParam,
    analysis_info: AnalysisInfo,
    existing_dc: Optional[DataContainer] = None,
    condition: Optional[str] = None,
    expt_metric_df_materialize_type: str = "Query",
    create_db_table_specs: Optional[CreateDBTableSpecs] = None,
) -> DataContainer:
    """This is the flow to get data for multi-experiment analysis (MEA),
    returning a DataContainer.

    Steps:

    1. Get the experiment assignment data.
    2. Get the metric data.
    3. Join the experiment assignment and metric data.
        The join is done on the unit column of the experiment assignment data.
        `expt_unit_col` is extracted from `expt_info.expt_unit_col`.
        and `metric_join_unit_col` is extracted from `metric_family.metric_join_unit_col`.
    4. Process the data.

    Args:
        io_param: IO parameters to read / write tables and results. This includes a cursor (`Cursor` type) to interact with
            database / read write data. Also it includes other arguments such as `create_table_prefix` to facilitate writing tables
            if needed and apply the same parameter to all tables generated with the flow.
        analysis_info: `AnalysisInfo` which includes the experiments and metrics.
        expt_asgmnt_table: Table which includes expt data.
        get_asgmnt_query: Callable which creates expt query.
        existing_dc: If such DataContainer exists, we augment more metrics to it.
            In particular we will assume there is no need to query expt assignment data.
        condition: Optional condition for sql query, eg to subsample.
        expt_metric_df_materialize_type: How to materialize the data. Options are "Pandas", "Query", "DBTable".
            This is the meaning of each type:

            - "Pandas": The resulting DataContainer will include a pandas dataframe.
            - "Query": The resulting DataContainer will include a query string to get the data.
            - "DBTable": The resulting DataContainer will include a table name which includes the data.

        create_db_table_specs: This specifies how to write DB table.
            See `CreateDBTableSpecs`.

    Returns:
        result: Processed DataContainer for MEA.

    Alters:
        analysis_info.multi_expt_info: This function will amend `multi_expt_info` by adding `derived_stats` field.
            It will add `derived_stats` also to all  `expt_info` in the `expt_info_list`.

    """
    if expt_metric_df_materialize_type not in EXPT_METRIC_DF_MATERIALIZE_TYPES:
        raise ValueError(
            f"expt_metric_df_materialize_type {expt_metric_df_materialize_type} not supported. " f"Supported types: {EXPT_METRIC_DF_MATERIALIZE_TYPES}"
        )

    if create_db_table_specs is None:
        create_db_table_specs = CreateDBTableSpecs(
            table_name=None,
            table_name_core=None,
            add_date_suffix=True,
            # add_date_suffix_format="%Y%m%d",
            # drop_if_exists=True,
            # dialect="Trino",
            # retries=1,
        )

    cursor = io_param.cursor
    multi_expt_info = analysis_info.multi_expt_info
    # Takes intersection from time periods of all experiments
    # This will define the analysis period
    if analysis_info.start_date is None:
        analysis_info.start_date = max([expt_info.start_date for expt_info in multi_expt_info.expt_info_list])

    if analysis_info.end_date is None:
        analysis_info.end_date = min([expt_info.end_date for expt_info in multi_expt_info.expt_info_list])

    print(f"\n*** analysis_info start date and end dates after taking intersections: {analysis_info.start_date}, {analysis_info.end_date}")

    # We then update the expt_infos as well to adhere to these dates.
    # Note that it is possible that expt_info has a tighter range for some experiments if desired.
    # However that would be a rare case.
    # The `min`, `max` below will ensure that the experiment range is not wider than the analysis range.
    # Also note that `analysis_info.start_date` and `analysis_info.end_date` are None, the ranges will be all the same.
    # This is because in the above these two quantities are calculated from `expt_info_list` in that case.
    for expt_info in multi_expt_info.expt_info_list:
        expt_info.start_date = max(analysis_info.start_date, expt_info.start_date)
        expt_info.end_date = min(analysis_info.end_date, expt_info.end_date)

    # If either of `expt_df` and `processed_df` are not None.
    # We assume that we only are supposed to join with more metrics.
    # Therefore we do not query the expt assignment data anymore.
    if existing_dc is None:
        existing_dc = get_multi_expt_data(
            cursor=cursor,
            multi_expt_info=multi_expt_info,
            condition=condition,
        )
        # Expt data

    print("\n*** multi_expt_info.derived_stats (after getting the data):\n" f"{multi_expt_info.derived_stats}")

    # Let us determine the expt_unit_col first
    if multi_expt_info.expt_unit_col is not None:
        expt_unit_col = multi_expt_info.expt_unit_col
        print("\n*** In `get_mea_data`: expt_unit_col was set " f"from multi_expt_info.expt_unit_col: {multi_expt_info.expt_unit_col}")
    elif expt_info.expt_unit_col is not None:
        expt_unit_col = expt_info.expt_unit_col
        print("\n*** In `get_mea_data`: expt_unit_col was set" f"from the last expt_info.expt_unit_col: {expt_info.expt_unit_col}")
    else:
        raise ValueError(
            "\n*** In `get_mea_data` in order to join expt assignment and metrics data, "
            "expt_unti_col must be available in multi_expt_info (MultiExptInfo) of the last expt_info (ExptInfo)."
        )

    # Below we keep augmenting the available data with more metrics.
    # This is done as long as `metric_info_list` is non-empty.
    # Otherwise, `existing_df` will only contain `expt_df`.
    # That would be still useful eg when we want to only return `expt_df`
    # and do joins with metric data in multiple steps.
    # The for loop will keep augmenting `existing_df`
    if analysis_info.metric_info_list:
        for metric_info in analysis_info.metric_info_list:
            metric_family = metric_info.metric_family
            metric_join_unit_col = metric_family.metric_join_unit_col

            process_expt_metric_df = metric_family.process_expt_metric_df
            process_expt_metric_df_params = metric_family.process_expt_metric_df_params

            # Use get_u_metrics_query_from_info to construct the metric query
            metric_query = get_u_metrics_query_from_info(
                metric_info=metric_info,
                start_date=analysis_info.start_date,
                end_date=analysis_info.end_date,
                condition=condition,
            )

            # Get u_metrics for later use in joins and processing
            u_metrics = get_u_metrics(metric_info.metrics)

            print(f"\n*** `metric_query`:\n{metric_query}")

            # Two cases here:
            # Case 1: existing_dc is a pandas dataframe:
            # In this case, we get the metrics data as dataframe first too and then join as pandas dataframes
            # Case 2: existing_dc does not include a df
            # In this case it should include a query / table_name instead
            # We will then will construct a join query with metrics data (u_metrics)
            # And then get the pandas data in one step.

            # First we do a sanity check to make sure that if `existing_dc` includes a pandas_df,
            # then `expt_metric_df_materialize_type` must be "Pandas".
            if existing_dc.pandas_df is not None and expt_metric_df_materialize_type != "Pandas":
                raise ValueError("If existing_dc includes a pandas_df, expt_metric_df_materialize_type must be 'Pandas'.")

            is_materialized_as_pandas = False

            if existing_dc.pandas_df is not None:
                metric_query_result = cursor.get_df(query=metric_query)
                metric_df = metric_query_result.df
                print(f"\n*** metric_df shape:\n{metric_df.shape}")
                print(f"\n*** metric_df:\n{metric_df.head(2)}")
                mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)  # Convert to GB
                print(f"***\n Memory used in `get_mea_data` after getting metric data: {mem_usage:.4f} GB")

                # TODO: add a check for repeated metric names.
                # This will be a left join as that is the default for the function.
                existing_dc = join_expt_with_metric_df(
                    expt_dc=existing_dc,
                    metric_dc=DataContainer(pandas_df=metric_df, is_pandas_df=True),
                    u_metrics=u_metrics,
                    expt_unit_col=expt_unit_col,
                    metric_join_unit_col=metric_join_unit_col,
                )

                mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)  # Convert to GB
                print("***\n Memory used in `get_mea_data` after joining `expt_df`, `metric_df`:" f" {mem_usage:.4f} GB")

                print(f"\n*** existing_dc:\n{existing_dc.df.head(2) if existing_dc.df is not None else None}")
                print("\n*** deletes data we do not need: `metric_df`")

                del metric_df
                del metric_query_result
                gc.collect()
                mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)  # Convert to GB
                print(f"***\n Memory used in `get_mea_data` after gc: {mem_usage:.4f} GB")
                is_materialized_as_pandas = True
            else:
                expt_query = existing_dc.table_name or existing_dc.query
                join_expt_metric_query = join_expt_with_metric_queries(
                    expt_assignment_query=expt_query,
                    metric_query=metric_query,
                    expt_unit_col=expt_unit_col,
                    metric_join_unit_col=metric_join_unit_col,
                    u_metrics=u_metrics,
                )

                existing_dc = DataContainer(query=join_expt_metric_query, pandas_df=None, is_pandas_df=False)
                print(f"\n*** existing_dc.query after joining with metrics:\n{existing_dc.query}")
        # End of for loop over metric_info_list
        # At this stage existing_dc includes all expt and metric data.
        # This could be in the form of a pandas_df, table_name, or query,
        # depending on how we materialized the data so far.
        # In the case that that initial existing_dc included a pandas_df,
        # we have already materialized as pandas inside the for loop above.
        # Otherwise, we have a chance to materialize now based on
        # `expt_metric_df_materialize_type`.

        # Now we handle materialization based on `expt_metric_df_materialize_type`
        # Case 1: In the case where `expt_metric_df_materialize_type` is "Pandas", if we have not already
        # materialized as pandas, we do it now.
        # Case 2: When expt_metric_df_materialize_type is "DBTable", we create a table from the query.
        # Case 3: When expt_metric_df_materialize_type is "Query", we do nothing as the query is already set.
        if expt_metric_df_materialize_type == "Pandas" and not is_materialized_as_pandas:
            join_expt_metric_query_result = cursor.get_df(query=existing_dc.query)
            df = join_expt_metric_query_result.df
            print(f"\n*** joined expt / metric df shape:\n{df.shape}")
            # Convert variant column to tuple (Trino returns NamedRowTuple, DuckDB returns dict)
            df[VARIANT_COL] = [tuple(item.values()) if isinstance(item, dict) else tuple(item) for item in df[VARIANT_COL]]
            print(f"\n*** joined df:\n{df.head(2)}")
            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)  # Convert to GB
            print(f"***\n Memory used in `get_mea_data` after getting joined df (via a join query): {mem_usage:.4f} GB")
            is_materialized_as_pandas = True
            existing_dc = DataContainer(pandas_df=df, is_pandas_df=True)
            print("\n*** In `get_mea_data` data was materialized as pandas_df.")
        elif expt_metric_df_materialize_type == "DBTable":
            add_table_name_if_needed(
                dc=existing_dc,
                create_db_table_specs=create_db_table_specs,
                multi_expt_info=multi_expt_info,
            )
            existing_dc.materialize_to_db_table(
                io_param=io_param,
                create_db_table_specs=create_db_table_specs,
            )
            print(f"\n*** In `get_mea_data`: Table {create_db_table_specs.table_name} created successfully for MEA data.")
            print("\n*** In `get_mea_data`: No pandas df was materialized here.")
            existing_dc.is_sql_df = True
        elif expt_metric_df_materialize_type == "Query":
            # Nothing to do, existing_dc already has the query
            print("\n*** In `get_mea_data`: expt_metric_df_materialize_type='Query'. " + "Setting is_pandas_df, is_sql_df to False.")
            existing_dc.is_sql_df = False
            existing_dc.is_pandas_df = False
            existing_dc.table_name = None

        # Processing the final pansdas_df if applicable
        if process_expt_metric_df is not None and existing_dc.pandas_df is not None:
            existing_dc.pandas_df = process_expt_metric_df(df=existing_dc.pandas_df, u_metrics=u_metrics, **process_expt_metric_df_params)

    return existing_dc


def get_mea_data_and_materialize(
    io_param: IOParam,
    analysis_info: AnalysisInfo,
    existing_dc: Optional[DataContainer] = None,
    condition: Optional[str] = None,
    expt_metric_df_materialize_type: str = "Query",
    expt_metric_df_materialize_spec=("DBTable",),
    create_db_table_specs: Optional[CreateDBTableSpecs] = None,
) -> DataContainer:
    """See `get_mea_data` for args and return. The only new argument is:

    Args:
        expt_metric_df_materialize_type: The default is "Query" as this involves no computation.
            The matrialization is pushed down to later.
        expt_metric_df_materialize_spec: Spec to determine how to materialize the data.
            See `get_mea_data_and_materialize` for more details. Acceptable values are

                - ("Pandas",): pandas dataframe
                - ("DBTable,): DB table
                - ("DBTable, "Pandas"): both materizalize to DB and create pandas
                - None: will not do anything beyond populating .query field in dc.
        create_db_table_specs: This specifies how to write DB table.
            See `CreateDBTableSpecs`.
    """

    if expt_metric_df_materialize_spec not in EXPT_METRIC_DF_MATERIALIZE_SPECS:
        raise ValueError(
            f"In `get_mea_data_and_materialize`, expt_metric_df_materialize_spec: {expt_metric_df_materialize_spec} "
            "needs to be one of EXPT_METRIC_DF_MATERIALIZE_SPECS. "
            f"Supported types: {EXPT_METRIC_DF_MATERIALIZE_SPECS}. "
            "Notice that the order in the tuple matters as well."
        )

    dc = get_mea_data(
        io_param=io_param,
        analysis_info=analysis_info,
        existing_dc=existing_dc,
        condition=condition,
        expt_metric_df_materialize_type=expt_metric_df_materialize_type,
    )

    # We need to initialize create_db_table_specs if not passed.
    if create_db_table_specs is None:
        create_db_table_specs = CreateDBTableSpecs(
            table_name=None,
            table_name_core=None,
            add_date_suffix=True,
            # add_date_suffix_format="%Y%m%d",
            # drop_if_exists=True,
            # dialect="Trino",
            # retries=1,
        )

    # It is more optimal to first materizalie to DB if DBTable is in specs
    # This is because The Pandas DF then can be directly extracted from a prepped table below
    # This happens because `materialize_to_pandas_df` will use the table_name if it finds on in the DC.
    if "DBTable" in expt_metric_df_materialize_spec:
        add_table_name_if_needed(
            dc=dc,
            create_db_table_specs=create_db_table_specs,
            multi_expt_info=analysis_info.multi_expt_info,
        )
        dc.materialize_to_db_table(io_param=io_param, create_db_table_specs=create_db_table_specs)
        print("\n***: In `get_mea_data_and_materialize`, DBTable was materialized.")
        dc.is_sql_df = True

    if "Pandas" in expt_metric_df_materialize_spec:
        dc.materialize_to_pandas_df(cursor=io_param.cursor)
        # Processing the final pansdas_df if applicable
        # Note that we skip applying `process_expt_metric_df` here
        # This is because `process_expt_metric_df` is supposed to be defined per metric_family.
        # If one wants to use that feature, one needs to materialize each metric_family to pandas
        # in get_mea_data

    return dc


def add_table_name_if_needed(dc: DataContainer, create_db_table_specs: CreateDBTableSpecs, multi_expt_info: MultiExptInfo):
    """This is a helper function which will create a table_name_core based on expt names if no table_name is
    given in DC or CreateDBTableSpecs.

    Args:
        dc: DataContainer for MEA.
        multi_expt_info: `MultiExptInfo` which includes the experiments and metrics.
        create_db_table_specs: This specifies how to write DB table.
            See `CreateDBTableSpecs`.
        expt_asgmnt_table: Table which includes expt data.

    Alters:
        create_db_table_specs (CreateDBTableSpecs)
    """

    if create_db_table_specs.table_name is None and create_db_table_specs.table_name_core is None and dc.table_name is None:
        # If there is no table_name or table_name_core is specified at all we generate one
        table_name_core = "mea_data"
        if multi_expt_info.expt_info_list:
            expt_names = [expt_info.name for expt_info in multi_expt_info.expt_info_list]
            table_name_core += "_" + "_".join(expt_names[:20])  # Max 20 expts in name
            table_name_core = table_name_core[:100]  # Limit length to 100 chars
            temp_io_param = IOParam()
            table_name_core = temp_io_param._sanitize_date_string(table_name_core)
        print(f"In `get_mea_data_and_materialize` table_name_core was generated: {table_name_core}")
        create_db_table_specs.table_name_core = table_name_core
