"""
Title: Database
Date: 26/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

# Initialise Connection Snippet
import sys
from pathlib import Path
import psycopg2
from psycopg2.extras import DictCursor, NamedTupleCursor
from psycopg2 import OperationalError, errorcodes, errors
from collections import namedtuple
import traceback
import streamlit as st
from typing import List
# from config import config
# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
# TEST_MODULE_PATH = SRC / "test" / "test_page" / "module"

for path in sys.path:
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))  # ./lib
    else:
        pass

    # if str(TEST_MODULE_PATH) not in sys.path:
    #     sys.path.insert(0, str(TEST_MODULE_PATH))
    # else:
    #     pass
# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger


# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# dsn = "host=localhost port=5432 dbname=eye user=shrdc password=shrdc"

# Initialise Connection to PostgreSQL Database Server

def print_psycopg2_exception(err):
    # get details about the exception
    err_type, err_obj, traceback = sys.exc_info()

    # get the line number when exception occured
    line_num = traceback.tb_lineno
    # print the connect() error
    print("\npsycopg2 ERROR:", err, "on line number:", line_num)
    print("psycopg2 traceback:", traceback, "-- type:", err_type)

    # psycopg2 extensions.Diagnostics object attribute
    print("\nextensions.Diagnostics:", err.diag)

    # print the pgcode and pgerror exceptions
    print("pgerror:", err.pgerror)
    print("pgcode:", err.pgcode, "\n")


def test_db_conn(dsn):
    try:
        conn = psycopg2.connect(dsn)

    except Exception as e:
        log_error(e)
        print_psycopg2_exception(e)
        st.exception(e)
        conn = None
    return conn


@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_connection(dsn=None, connection_factory=None, cursor_factory=None, **kwargs):
    """ Connect to the PostgreSQL database server """

    try:
        # read connection parameters
        # params = config()

        # connect to the PostgreSQL server
        log_info('Connecting to the PostgreSQL database...')
        try:
            if kwargs:
                conn = psycopg2.connect(**kwargs)

            else:
                conn = psycopg2.connect(
                    dsn, connection_factory, cursor_factory)

        except Exception as e:
            log_error(e)
            print_psycopg2_exception(e)
            conn = None
        # create a cursor
        with conn:
            with conn.cursor(cursor_factory=NamedTupleCursor) as cur:

                # execute a statement
                cur.execute('SELECT version();')
                conn.commit()

                # display the PostgreSQL database server version
                db_version = cur.fetchone().version
                log_info(f"PostgreSQL database version: {db_version}")
                log_info(f"PostgreSQL connection status: {conn.info.status}")
                log_info(
                    f"You are connected to database '{conn.info.dbname}' as user '{conn.info.user}' on host '{conn.info.host}' at port '{conn.info.port}'.")

    except (Exception, psycopg2.DatabaseError) as error:
        log_error(error)
        conn = None

    return conn
    # finally:
    #     if conn is not None:
    #         conn.close()
    #         print('Database connection closed.')


def db_no_fetch(sql_message: str, conn, vars: List = None) -> None:
    with conn:
        with conn.cursor() as cur:
            try:
                if vars:
                    cur.execute(sql_message, vars)
                else:
                    cur.execute(sql_message)
                conn.commit()
            except psycopg2.Error as e:
                log_error(e)


def db_fetchone(sql_message: str, conn, vars: List = None, fetch_col_name: bool = False, return_dict: bool = False) -> namedtuple:

    with conn:
        cursor_factory = DictCursor if return_dict else NamedTupleCursor

        with conn.cursor(cursor_factory=cursor_factory) as cur:

            try:

                if vars:
                    cur.execute(sql_message, vars)

                else:
                    cur.execute(sql_message)

                conn.commit()
                return_one = cur.fetchone()  # return tuple
                # Obtain Column names from query
                column_names = [desc[0] for desc in cur.description]

                if fetch_col_name:
                    if return_dict:
                        # Convert results to pure Python dictionary
                        return_one = convert_to_dict(return_one)
                    return return_one, column_names
                else:
                    if return_dict:
                        # Convert results to pure Python dictionary
                        return_one = convert_to_dict(return_one)
                    column_names = None
                    return return_one
            except psycopg2.Error as e:
                log_error(e)


def db_fetchall(sql_message: str, conn, vars: List = None, fetch_col_name: bool = False, return_dict: bool = False) -> namedtuple:
    with conn:
        cursor_factory = DictCursor if return_dict else NamedTupleCursor
        with conn.cursor(cursor_factory=cursor_factory) as cur:
            try:
                if vars:
                    cur.execute(sql_message, vars)
                else:
                    cur.execute(sql_message)
                conn.commit()
                return_all = cur.fetchall()  # return array of tuple
                column_names = [desc[0] for desc in cur.description]
                if fetch_col_name:
                    if return_dict:
                        # Convert results to pure Python dictionary
                        return_all = convert_to_dict(return_all)
                    return return_all, column_names
                else:
                    if return_dict:
                        return_all = convert_to_dict(return_all)
                    column_names = None
                    return return_all
            except psycopg2.Error as e:
                log_error(e)


def convert_to_dict(query: List):
    query_dict = [dict(row) for row in query]
    return query_dict
