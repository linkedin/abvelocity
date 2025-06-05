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

import prestodb

from abvelocity.get_data.cursor import Cursor


@dataclass
class PrestoConnArgs:
    """This is a dataclass to store Presto / Trino connection arguments e.g. host, port etc."""

    host: str
    """Host path, which is where the date resides."""
    user: Optional[str] = None
    """user name."""
    authorization_user: Optional[str] = None
    """authorization user."""
    authorization_user_specifier: Optional[str] = "li_authorization_user"
    """The string used in the specific SQL database to specify authorization user.
        By default this is `li_authorization_user` which is then used by Presto to set authorization_user.
        For example if authorization_user is "hypothetical-auth-user", then Pretso will run:
        `SET SESSION li_authorization_user = 'hypothetial-auth-user'`."""
    catalog: Optional[str] = "hive"
    port: Optional[int] = 443


class PrestoCursor(Cursor):
    """A Presto-specific cursor that inherits from the generic Cursor class."""

    def __init__(self, data_source: PrestoConnArgs):
        """Initializes the Presto cursor with connection parameters."""
        super().__init__(data_source)
        self._presto_cursor = self._create_presto_cursor()

        if data_source.authorization_user is not None:
            print("\n***setting `authorization_user`")
            auth_query = f"SET SESSION {data_source.authorization_user_specifier} = '{data_source.authorization_user}'"
            print(auth_query)
            self._presto_cursor.execute(auth_query)
            self._presto_cursor.fetchall()

    def _create_presto_cursor(self):
        """Creates and returns a Presto cursor."""
        if self.data_source.user is None:
            self.data_source.user = getpass.getuser()
            print(f"\n***user was inferred: user = {self.data_source.user}")

        print("Type password + 2FA (without spaces) and hit enter.")
        pw = getpass.getpass("Password + 2FA (without spaces): ")

        conn = prestodb.dbapi.connect(
            host=self.data_source.host,
            port=self.data_source.port,
            user=self.data_source.user,
            catalog=self.data_source.catalog,
            http_scheme="https",
            auth=prestodb.auth.BasicAuthentication(self.data_source.user, pw),
        )
        return conn.cursor()

    def execute(self, query: str):
        """Executes a Presto query."""
        print(f"\n*** executing query:\n{query}")
        self._presto_cursor.execute(query)
        self.fetchall()
        self.description = self._presto_cursor.description
        print(f"self.description: {self.description}")

    def fetchall(self):
        """Fetches all rows of a query result as a list of tuples or lists."""
        self.results = self._presto_cursor.fetchall()

    def fetchone(self):
        """Fetches the next row of a query result set."""
        return self._presto_cursor.fetchone()
