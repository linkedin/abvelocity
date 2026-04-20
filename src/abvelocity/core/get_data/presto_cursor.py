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


import getpass
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import trino
from abvelocity.core.get_data.cursor import DEFAULT_MAX_RETRIES, DEFAULT_RETRY_DELAY_SECONDS, Cursor
from trino.auth import OAuth2Authentication

# Increase the polling attempts so there is enough time to complete SSO in the browser.
# The default of 5 is too low for Microsoft Entra SSO + MFA flows.
# Wrapped in try/except in case the private class is renamed/removed in a future trino release.
try:
    from trino.auth import _OAuth2TokenBearer

    _OAuth2TokenBearer.MAX_OAUTH_ATTEMPTS = 60
except Exception:
    pass


@dataclass
class ConnArgs:
    """This is a dataclass to store Presto / Trino connection arguments e.g. host, port etc."""

    host: str
    """Host path, which is where the date resides."""
    user: Optional[str] = None
    """user name."""
    authorization_user: Optional[str] = None
    """authorization user."""
    authorization_user_specifier: Optional[str] = "auth_user_session_var"
    """The string used in the specific SQL database to specify authorization user.
        By default this is `auth_user_session_var` which is then used by Presto to set authorization_user.
        For example if authorization_user is "hypothetical-auth-user", then Pretso will run:
        `SET SESSION auth_user_session_var = 'hypothetial-auth-user'`."""
    catalog: Optional[str] = "hive"
    port: Optional[int] = 443
    schema: Optional[str] = "default"


class PrestoCursor(Cursor):
    """A Presto-specific cursor that inherits from the generic Cursor class."""

    def __init__(self, conn_args: ConnArgs, **kwargs):
        """Initializes the Presto cursor with connection parameters."""

        # Pass conn_args attributes to Cursor.__init__ (for backward compatibility)
        # Use kwargs to allow explicit setting of max_retries/retry_delay_seconds
        super().__init__(
            conn_args,
            max_retries=getattr(conn_args, "max_retries", DEFAULT_MAX_RETRIES),
            retry_delay_seconds=getattr(conn_args, "retry_delay_seconds", DEFAULT_RETRY_DELAY_SECONDS),
            **kwargs,
        )
        # Assign the Trino cursor to the base class's internal _cursor attribute
        self._cursor = self._create_presto_cursor()
        print("\n***: presto cursor was created.")

        if conn_args.authorization_user is not None:
            print("\n***setting `authorization_user`")
            auth_query = f"SET SESSION {conn_args.authorization_user_specifier} = '{conn_args.authorization_user}'"
            print(f"\n*** auth_query: {auth_query} was run on Trino.")

            # Use the inherited execute method for retries
            self.execute(auth_query)

    def _create_presto_cursor(self):
        """Creates and returns a Presto cursor."""
        if self.conn_args.user is None:
            self.conn_args.user = getpass.getuser()
            print(f"\n***user was inferred: user = {self.conn_args.user}")

        print("A browser window will open for SSO login when you first execute a query.")
        conn = trino.dbapi.connect(
            host=self.conn_args.host,
            port=self.conn_args.port,
            user=self.conn_args.user,
            catalog=self.conn_args.catalog,
            schema=self.conn_args.schema,
            http_scheme="https",
            auth=OAuth2Authentication(),
        )
        return conn.cursor()

    # --- IMPLEMENTATION OF ABSTRACT METHOD ---
    def _execute_core(self, query: str):
        """
        Implements the core execution logic for Trino (required by Cursor base class).
        This method will be wrapped by the parent's retry logic.
        """
        # Determine if this is the authorization query for simpler logging
        auth_query_template = f"SET SESSION {self.conn_args.authorization_user_specifier} = '{self.conn_args.authorization_user}'"
        is_auth_query = query == auth_query_template

        try:
            # 1. Execute the query using the internal cursor
            self._cursor.execute(query)

            # 2. Update description immediately after successful execution
            self.description = self._cursor.description

            if is_auth_query:
                # For the internal init query, we force a fetchall to ensure command completes
                self._cursor.fetchall()
                print(f"***: init query {query} succeeded.")
            else:
                # Standard query logging
                print(f"self.description: {self.description}")

        except Exception as e:
            # Contextual logging before the parent's retry loop catches the exception
            if not is_auth_query:
                print(f"*** Trino specific error: Most likely infra flakiness: {e}")
            else:
                print(f"*** Init query failed with Trino error: {e}")

            # Raise the exception for the parent's _execute_with_retries to handle the retry/failure.
            raise e

    # --- IMPLEMENTATION OF ABSTRACT FETCH METHODS ---
    def fetchall(self):
        """Fetches all rows of a query result as a list of tuples or lists."""
        if self._cursor is None:
            raise ValueError("Cursor is not initialized.")
        return self._cursor.fetchall()

    def fetchone(self):
        """Fetches the next row of a query result set."""
        if self._cursor is None:
            raise ValueError("Cursor is not initialized.")
        return self._cursor.fetchone()

    def write_pandas_df(
        self,
        df: "pd.DataFrame",
        table_name: str,
        mode: str = "append",
    ) -> None:
        """Not yet implemented for Presto/Trino.

        Trino has no native pandas write path.  The recommended approach is
        to write the DataFrame to parquet on shared storage and point a Trino
        external table at it.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "write_pandas_df is not supported for PrestoCursor. "
            "Write the DataFrame to parquet on shared storage and register "
            "it as a Trino external table instead."
        )
