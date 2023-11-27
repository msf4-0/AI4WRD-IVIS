"""
Title: Database
Date: 26/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)

Copyright (C) 2021 Selangor Human Resource Development Centre

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. 

Copyright (C) 2021 Selangor Human Resource Development Centre
SPDX-License-Identifier: Apache-2.0
========================================================================================

"""

# Initialise Connection Snippet
import os
import sys
from pathlib import Path
import psycopg2
from psycopg2.extras import DictCursor, NamedTupleCursor
from psycopg2 import sql, extensions
from collections import namedtuple
import traceback
from enum import IntEnum
from passlib.hash import argon2
from typing import Any, Dict, List, NamedTuple, Tuple, Union
import streamlit as st
from streamlit import session_state

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
from core.utils.log import logger  # logger
from core.utils.model_details_db_setup import check_if_pretrained_models_exist, connect_db, scrape_setup_model_details

# POSTGRES_DB env variable could be used by Docker for a different name
APP_DATABASE_NAME = os.environ.get(
    "POSTGRES_DB", "integrated_vision_inspection_system")


# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<
class DatabaseStatus(IntEnum):
    NotExist = 0
    Exist = 1

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return DatabaseStatus[s]
        except KeyError:
            raise ValueError()
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


def test_db_conn(dsn=None, connection_factory=None, cursor_factory=None, **kwargs):
    """Test connection to PostgreSQL database

    Args:
        dsn (array, optional): Database source name. Defaults to None.
        connection_factory (Callable, optional):  Defaults to None.
        cursor_factory (psycopg2.extras, optional): Data type of query results. Defaults to None.

    Returns:
        psycopg2.connection: connection object 
    """
    try:
        if kwargs:
            conn = psycopg2.connect(**kwargs)
        else:
            conn = psycopg2.connect(dsn, connection_factory, cursor_factory)

    except Exception as e:
        logger.error(e)
        print_psycopg2_exception(e)
        conn = None
    return conn


@st.experimental_singleton
# testing this new decorator above
# @st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_connection(dsn=None, connection_factory=None, cursor_factory=None, **kwargs):
    """ Connect to the PostgreSQL database server """

    try:
        # read connection parameters
        # params = config()

        # connect to the PostgreSQL server
        logger.info('Connecting to the PostgreSQL database...')
        try:
            if kwargs:
                logger.debug(f"Connection kwargs: {kwargs}")
                conn = psycopg2.connect(**kwargs)

            else:
                conn = psycopg2.connect(
                    dsn, connection_factory, cursor_factory)

        except Exception as e:
            logger.error(e)
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
                logger.info(f"PostgreSQL database version: {db_version}")
                logger.info(
                    f"PostgreSQL connection status: {conn.info.status}")
                logger.info(
                    f"You are connected to database '{conn.info.dbname}' as user '{conn.info.user}' on host '{conn.info.host}' at port '{conn.info.port}'.")

    except (Exception, psycopg2.DatabaseError) as error:
        logger.error(error)
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
                logger.error(e)


def db_fetchone(sql_message: str, conn, vars: List = None,
                fetch_col_name: bool = False,
                return_dict: bool = False
                ) -> Union[Union[NamedTuple, Dict[str, Any]],
                           Tuple[Union[NamedTuple, Dict[str, Any]],
                                 List[str]]]:
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
                if return_dict:
                    # Convert results to pure Python dictionary IF exists
                    if return_one:
                        return_one = dict(return_one)

                if fetch_col_name:
                    # Obtain Column names from query
                    column_names = [desc[0] for desc in cur.description]
                    return return_one, column_names
                else:
                    return return_one
            except psycopg2.Error as e:
                logger.error(e)


def db_fetchall(sql_message: str, conn, vars: List = None,
                fetch_col_name: bool = False,
                return_dict: bool = False
                ) -> Union[Union[List[NamedTuple], List[Dict[str, Any]]],
                           Tuple[Union[List[NamedTuple], List[Dict[str, Any]]],
                                 List[str]]]:
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
                if return_dict:
                    # Convert results to pure Python dictionary
                    return_all = convert_to_dict(return_all)

                if fetch_col_name:
                    column_names = [desc[0] for desc in cur.description]
                    return return_all, column_names
                else:
                    return return_all
            except psycopg2.Error as e:
                logger.error(e)


def convert_to_dict(query: List) -> Dict[str, Any]:
    query_dict = [dict(row) for row in query]
    return query_dict


def check_if_database_exist(datname: str, conn) -> bool:
    """Check if Database exists in the current connection

    Args:
        datname (str): Database name
        conn (psycopg2.connection): Connection object to the PostgreSQL server

    Returns:
        bool: True if exists, False otherwise
    """
    check_if_database_exist_SQL = """
    SELECT EXISTS
        (SELECT datname FROM pg_database WHERE datname = %s);
    """

    check_if_database_exist_vars = [datname]

    try:
        query_return = db_fetchone(
            check_if_database_exist_SQL, conn, check_if_database_exist_vars)
        exist_flag = query_return.exists
        logger.info(f"Database {datname} exists")

    except TypeError as e:
        logger.error(e)
        exist_flag = False

    return exist_flag


def check_if_table_exist(tablename: str, conn) -> bool:
    sql_query = """
    SELECT EXISTS
        (SELECT * FROM %s);
    """

    sql_vars = (extensions.AsIs(tablename),)

    query_return = db_fetchone(sql_query, conn, sql_vars)
    if query_return is not None:
        logger.info(f"Table {tablename} exists")
        exist_flag = True
    else:
        exist_flag = False
    return exist_flag

# ************************************ SETUP DATABASE *********************************************************#


def create_database(database_name: str, conn, owner: str = None):
    """Create Database

    Args:
        database_name (str): Database Name
        conn (psycopg2.connection): Connection object for PostgreSQL server
    """
    try:
        # Set Transaction Isolation Level to Autocommit: https://www.postgresql.org/docs/9.5/transaction-iso.html
        # or else it will throw error such as ".... cannot run inside a transaction block"
        autocommit = extensions.ISOLATION_LEVEL_AUTOCOMMIT
        conn.set_isolation_level(autocommit)

        with conn.cursor() as cur:
            # Create database using sql.Identifier to avoid SQL Injection via string
            logger.info(f"Creating Database: {database_name} OWNER {owner}")
            if owner:
                # Create Database with owner
                cur.execute(sql.SQL(
                    "CREATE DATABASE {} OWNER {};"
                ).format(sql.Identifier(database_name), sql.Identifier(owner)))
            else:
                # Create Database with postgres as owner
                cur.execute(sql.SQL(
                    "CREATE DATABASE {};"
                ).format(sql.Identifier(database_name)))

    except psycopg2.Error as e:
        logger.error(e)


def create_relation_database(conn):
    """Create Relation table in the Database

    Args:
        conn (psycopg2.connection): Connection object for PostgreSQL
    """
    # TODO #22 Password Hashing check
    initialise_database_SQL = """
        BEGIN;

        --
        -- Name: trigger_update_timestamp(); Type: FUNCTION; Schema: public; Owner:
        --
        CREATE OR REPLACE FUNCTION trigger_update_timestamp ()
            RETURNS TRIGGER
            AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$
        LANGUAGE plpgsql;

        -- ACCOUNT_STATUS table ---------------------------
        CREATE TABLE IF NOT EXISTS public.account_status (
            id integer NOT NULL GENERATED ALWAYS AS IDENTITY
            , name character varying(50) UNIQUE NOT NULL
            , PRIMARY KEY (id)
        );

        INSERT INTO public.account_status (
            name)
        VALUES (
            'NEW')
        , (
            'ACTIVE')
        , (
            'LOCKED')
        , (
            'LOGGED_IN')
        , (
            'LOGGED_OUT');

        -- ANNOTATIONS table --------------------------------------------------
        CREATE TABLE IF NOT EXISTS public.annotations (
            id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
            MINVALUE 1
            MAXVALUE 9223372036854775807
            CACHE 1)
            , result jsonb[]
            , annotation_type_id integer
            , project_id bigint NOT NULL
            , users_id bigint NOT NULL
            , task_id bigint NOT NULL
            , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , PRIMARY KEY (id)
        );

        CREATE TRIGGER annotations_update
            BEFORE UPDATE ON public.annotations
            FOR EACH ROW
            EXECUTE PROCEDURE trigger_update_timestamp ();

        -- DATASET table --------------------------------------------------
        CREATE TABLE IF NOT EXISTS public.dataset (
            id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
            MINVALUE 1
            MAXVALUE 9223372036854775807
            CACHE 1)
            , name text NOT NULL UNIQUE
            , description text
            , dataset_size integer
            , filetype_id integer
            , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , PRIMARY KEY (id)
        );

        CREATE TRIGGER dataset_update
            BEFORE UPDATE ON public.dataset
            FOR EACH ROW
            EXECUTE PROCEDURE trigger_update_timestamp ();

        --  DEPLOYMENT_TYPE table --------------------------------------------------
        CREATE TABLE IF NOT EXISTS public.deployment_type (
            id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
            MINVALUE 1
            MAXVALUE 9223372036854775807
            CACHE 1)
            , name character varying(100) NOT NULL
            , template TEXT
            , PRIMARY KEY (id)
        );

        INSERT INTO public.deployment_type (
            name)
        VALUES (
            'Image Classification')
        , (
            'Object Detection with Bounding Boxes')
        , (
            'Semantic Segmentation with Polygons');

        -- EDITOR table --------------------------------------------------
        CREATE TABLE IF NOT EXISTS public.editor (
            id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
            MINVALUE 1
            MAXVALUE 9223372036854775807
            CACHE 1)
            , name character varying(50) NOT NULL
            , editor_config text
            , labels jsonb
            , project_id bigint NOT NULL
            , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , PRIMARY KEY (id)
        );

        CREATE TRIGGER editor_update
            BEFORE UPDATE ON public.editor
            FOR EACH ROW
            EXECUTE PROCEDURE trigger_update_timestamp ();

        -- FILETYPE table ---------------------------
        CREATE TABLE IF NOT EXISTS public.filetype (
            id integer NOT NULL GENERATED ALWAYS AS IDENTITY
            , name character varying(50) UNIQUE NOT NULL
            , PRIMARY KEY (id)
        );

        INSERT INTO public.filetype (
            name)
        VALUES (
            'Image') ,
        --jpeg,jpg,png
        (
            'Video') ,
        -- mp4,mpeg,webm*,ogg*
        (
            'Audio') ,
        --* wav, aiff, mp3, au, flac, m4a, ogg
        (
            'Text') -- txt,csv,tsv,json*,html*
        ;

        -- FRAMEWORK table --------------------------------------------------
        CREATE TABLE IF NOT EXISTS public.framework (
            id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
            MINVALUE 1
            MAXVALUE 9223372036854775807
            CACHE 1)
            , name character varying(50) NOT NULL
            , link text
            , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , PRIMARY KEY (id)
        );

        CREATE TRIGGER framework_update
            BEFORE UPDATE ON public.framework
            FOR EACH ROW
            EXECUTE PROCEDURE trigger_update_timestamp ();

        INSERT INTO public.framework (
            name
            , link)
        VALUES (
            'TensorFlow'
            , 'https://www.tensorflow.org/')
        , (
            'PyTorch'
            , 'https://pytorch.org/')
        , (
            'Scikit-learn'
            , 'https://scikit-learn.org/stable/')
        , (
            'Caffe'
            , 'https://caffe.berkeleyvision.org/')
        , (
            'MXNet'
            , 'https://mxnet.apache.org/')
        , (
            'ONNX'
            , 'https://onnx.ai/');

        -- MODEL_TYPE table --------------------------------------------------
        CREATE TABLE IF NOT EXISTS public.model_type (
            id smallint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
            MINVALUE 1
            MAXVALUE 32767
            CACHE 1)
            , name character varying(50) NOT NULL
            , PRIMARY KEY (id)
        );

        INSERT INTO public.model_type (
            name)
        VALUES (
            'Pre-trained Models')
        , (
            'Project Models')
        , (
            'User Deep Learning Model Upload');

        -- MODELS table --------------------------------------------------
        CREATE TABLE IF NOT EXISTS public.models (
            id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
            MINVALUE 1
            MAXVALUE 9223372036854775807
            CACHE 1)
            , name text NOT NULL UNIQUE
            , description text
            , metrics jsonb
            ,
            /*to store evaluation metrics for model such as COCO mAP and Training Losses*/
            model_path text
            , model_type_id smallint
            , framework_id bigint
            , deployment_id integer
            , training_id bigint
            , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , PRIMARY KEY (id)
        );

        CREATE TRIGGER models_update
            BEFORE UPDATE ON public.models
            FOR EACH ROW
            EXECUTE PROCEDURE trigger_update_timestamp ();

        -- PREDICTIONS table --------------------------------------------------
        CREATE TABLE IF NOT EXISTS public.predictions (
            id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
            MINVALUE 1
            MAXVALUE 9223372036854775807
            CACHE 1)
            , result jsonb[]
            , task_id bigint NOT NULL
            , model_id bigint
            , pre_trained_model_id bigint
            , score double precision
            , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , PRIMARY KEY (id)
        );

        CREATE TRIGGER predictions_update
            BEFORE UPDATE ON public.predictions
            FOR EACH ROW
            EXECUTE PROCEDURE trigger_update_timestamp ();

        --  PROJECT table --------------------------------------------------
        CREATE TABLE IF NOT EXISTS public.project (
            id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY (INCREMENT 1 START 1
            MINVALUE 1
            MAXVALUE 9223372036854775807
            CACHE 1)
            , name text NOT NULL UNIQUE
            , description text
            , deployment_id integer
            , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , PRIMARY KEY (id)
        );

        CREATE TRIGGER project_update
            BEFORE UPDATE ON public.project
            FOR EACH ROW
            EXECUTE PROCEDURE trigger_update_timestamp ();

        -- PROJECT_DATASET table (Many-to-Many) --------------------------------------------------
        CREATE TABLE IF NOT EXISTS public.project_dataset (
            project_id bigint NOT NULL
            , dataset_id bigint NOT NULL
            , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , PRIMARY KEY (project_id , dataset_id)
        );

        CREATE TRIGGER project_dataset_update
            BEFORE UPDATE ON public.project_dataset
            FOR EACH ROW
            EXECUTE PROCEDURE trigger_update_timestamp ();

        -- PROJECT_TRAINING table (Many-to-Many) ---------------------------
        CREATE TABLE IF NOT EXISTS public.project_training (
            project_id bigint NOT NULL
            , training_id bigint NOT NULL
            , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , PRIMARY KEY (project_id , training_id)
        );

        CREATE TRIGGER project_training_update
            BEFORE UPDATE ON public.project_training
            FOR EACH ROW
            EXECUTE PROCEDURE trigger_update_timestamp ();

        -- CHECK (account_status IN ('NEW', 'ACTIVE', 'LOCKED', 'LOGGED_IN', 'LOGGED_OUT'))
        --  ROLES table --------------------------------------------------
        CREATE TABLE IF NOT EXISTS public.roles (
            id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
            MINVALUE 1
            MAXVALUE 9223372036854775807
            CACHE 1)
            , name character varying(50) NOT NULL
            , page_access_list text[]
            , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , PRIMARY KEY (id)
        );

        CREATE TRIGGER roles_update
            BEFORE UPDATE ON public.roles
            FOR EACH ROW
            EXECUTE PROCEDURE trigger_update_timestamp ();

        -- INSERT pre-defined values into ROLES
        INSERT INTO public.roles (
            name
            , page_access_list)
        VALUES (
            'Administrator'
            , ARRAY['Login' , 'Project' , 'Dataset' , 'Editor' , 'Model Training' , 'Deployment' , 'User Management'])
        , (
            'Developer 1 (Deployment)'
            , ARRAY['Login' , 'Project' , 'Dataset' , 'Editor' , 'Model Training' , 'Deployment'])
        , (
            'Developer 2 (Model Training)'
            , ARRAY['Login' , 'Project' , 'Dataset' , 'Editor' , 'Model Training'])
        , (
            'Annotator'
            , ARRAY['Login' , 'Project' , 'Dataset' , 'Editor']);

        --  SESSION_LOG table --------------------------------------------------
        CREATE TABLE IF NOT EXISTS public.session_log (
            id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (CYCLE INCREMENT 1 START 1
            MINVALUE 1
            MAXVALUE 9223372036854775807
            CACHE 1)
            , users_id bigint NOT NULL
            , login_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , logout_at timestamp with time zone
            , PRIMARY KEY (id)
        );

        --  TASK table --------------------------------------------------
        CREATE TABLE IF NOT EXISTS public.task (
            id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY (INCREMENT 1 START 1
            MINVALUE 1
            MAXVALUE 9223372036854775807
            CACHE 1)
            , name text NOT NULL
            , dataset_id bigint NOT NULL
            , project_id bigint
            , annotation_id bigint
            , prediction_id bigint
            , is_labelled boolean NOT NULL DEFAULT FALSE
            , skipped boolean NOT NULL DEFAULT FALSE
            , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , PRIMARY KEY (id)
        );

        CREATE TRIGGER task_update
            BEFORE UPDATE ON public.task
            FOR EACH ROW
            EXECUTE PROCEDURE trigger_update_timestamp ();

        --  TRAINING table --------------------------------------------------
        CREATE TABLE IF NOT EXISTS public.training (
            id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
            MINVALUE 1
            MAXVALUE 9223372036854775807
            CACHE 1)
            , name text NOT NULL UNIQUE
            , description text
            , training_model_id bigint
            ,
            /* FK to 'model' table for model trained using current configuration */
            attached_model_id bigint
            ,
            /* FK to 'models' table for models attached to current training */
            training_param jsonb
            , augmentation jsonb
            , partition_ratio jsonb
            , is_started boolean NOT NULL DEFAULT FALSE
            , progress jsonb
            ,
            /* stores progress tracking of model training such as checkpoint and previous training steps*/
            created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , project_id bigint NOT NULL
            , PRIMARY KEY (id)
        );

        CREATE TRIGGER training_update
            BEFORE UPDATE ON public.training
            FOR EACH ROW
            EXECUTE PROCEDURE trigger_update_timestamp ();

        -- TRAINING_DATASET table (Many-to-Many) ---------------------------
        CREATE TABLE IF NOT EXISTS public.training_dataset (
            training_id bigint NOT NULL
            , dataset_id bigint NOT NULL
            , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , PRIMARY KEY (training_id , dataset_id)
        );

        CREATE TRIGGER training_dataset_update
            BEFORE UPDATE ON public.training_dataset
            FOR EACH ROW
            EXECUTE PROCEDURE trigger_update_timestamp ();

        -- TRAINING_LOG table --------------------------------------------------
        CREATE TABLE IF NOT EXISTS public.training_log (
            id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (CYCLE INCREMENT 1 START 1
            MINVALUE 1
            MAXVALUE 9223372036854775807
            CACHE 1)
            , users_id bigint NOT NULL
            , training_id bigint NOT NULL
            , start_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP ,
            --update using Python
            end_at timestamp with time zone NOT NULL
            , PRIMARY KEY (id)
        );

        -- USERS table ------------------------------------------------
        CREATE TABLE IF NOT EXISTS public.users (
            id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY (INCREMENT 1 START 1
            MINVALUE 1
            MAXVALUE 9223372036854775807
            CACHE 1)
            , emp_id text UNIQUE
            , username text NOT NULL UNIQUE
            , first_name text
            , last_name text
            , email text
            , department text
            , position text
            , psd text NOT NULL
            , roles_id integer NOT NULL
            , status_id integer NOT NULL DEFAULT 1
            , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
            , last_activity timestamp with time zone
            , PRIMARY KEY (id)
        );

        CREATE TRIGGER users_update
            BEFORE UPDATE ON public.users
            FOR EACH ROW
            EXECUTE PROCEDURE trigger_update_timestamp ();

        -- Default Admin Account
        /* 
        emp_id:0
        username:admin
        password:hash of 'admin'
        Role:Admininstrator */
        INSERT INTO public.users (
            emp_id
            , username
            , first_name
            , last_name
            , email
            , department
            , position
            , psd
            , roles_id)
        VALUES (
            '0'
            , 'admin'
            , NULL
            , NULL
            , NULL
            , NULL
            , NULL
            , %s
            , (
                SELECT
                    id
                FROM
                    public.roles
                WHERE
                    name = 'Administrator'));

        -----------------------------------Foreign Keys Constraint---------------------------------------------------
        -- USERS
        ALTER TABLE IF EXISTS public.users
            ADD CONSTRAINT fk_roles_id FOREIGN KEY (roles_id) REFERENCES public.roles (id) ON DELETE SET NULL NOT VALID;

        ALTER TABLE public.users VALIDATE CONSTRAINT fk_roles_id;

        ALTER TABLE IF EXISTS public.users
            ADD CONSTRAINT fk_status_id FOREIGN KEY (status_id) REFERENCES public.account_status (id) ON DELETE SET NULL ON UPDATE CASCADE NOT VALID;

        ALTER TABLE public.users VALIDATE CONSTRAINT fk_status_id;

        -- SESSION_LOG
        ALTER TABLE IF EXISTS public.session_log
            ADD CONSTRAINT fk_users_id FOREIGN KEY (users_id) REFERENCES public.users (id) ON DELETE CASCADE NOT VALID;

        ALTER TABLE public.session_log VALIDATE CONSTRAINT fk_users_id;

        -- PROJECT
        ALTER TABLE IF EXISTS public.project
            ADD CONSTRAINT fk_deployment_id FOREIGN KEY (deployment_id) REFERENCES public.deployment_type (id) ON DELETE SET NULL NOT VALID;

        ALTER TABLE public.project VALIDATE CONSTRAINT fk_deployment_id;

        -- TRAINING
        -- training model
        ALTER TABLE IF EXISTS public.training
            ADD CONSTRAINT fk_training_model_id FOREIGN KEY (training_model_id) REFERENCES public.models (id) ON DELETE SET NULL NOT VALID;

        ALTER TABLE public.training VALIDATE CONSTRAINT fk_training_model_id;

        -- attached model
        ALTER TABLE IF EXISTS public.training
            ADD CONSTRAINT fk_attached_model_id FOREIGN KEY (attached_model_id) REFERENCES public.models (id) ON DELETE SET NULL NOT VALID;

        ALTER TABLE public.training VALIDATE CONSTRAINT fk_attached_model_id;

        -- project id
        ALTER TABLE IF EXISTS public.training
            ADD CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE NOT VALID;

        ALTER TABLE public.training VALIDATE CONSTRAINT fk_project_id;

        -- TRAINING_LOG
        ALTER TABLE IF EXISTS public.training_log
            ADD CONSTRAINT fk_users_id FOREIGN KEY (users_id) REFERENCES public.users (id) ON DELETE NO ACTION NOT VALID;

        ALTER TABLE public.training_log VALIDATE CONSTRAINT fk_users_id;

        ALTER TABLE IF EXISTS public.training_log
            ADD CONSTRAINT fk_training_id FOREIGN KEY (training_id) REFERENCES public.training (id) ON DELETE NO ACTION NOT VALID;

        ALTER TABLE public.training_log VALIDATE CONSTRAINT fk_training_id;

        -- MODELS
        -- training id
        ALTER TABLE IF EXISTS public.models
            ADD CONSTRAINT fk_training_id FOREIGN KEY (training_id) REFERENCES public.training (id) ON DELETE SET NULL NOT VALID;

        ALTER TABLE public.models VALIDATE CONSTRAINT fk_training_id;

        -- framework id
        ALTER TABLE IF EXISTS public.models
            ADD CONSTRAINT fk_framework_id FOREIGN KEY (framework_id) REFERENCES public.framework (id) ON DELETE SET NULL NOT VALID;

        ALTER TABLE public.models VALIDATE CONSTRAINT fk_framework_id;

        -- deployment id
        ALTER TABLE IF EXISTS public.models
            ADD CONSTRAINT fk_deployment_id FOREIGN KEY (deployment_id) REFERENCES public.deployment_type (id) ON DELETE SET NULL NOT VALID;

        ALTER TABLE public.models VALIDATE CONSTRAINT fk_deployment_id;

        -- model type id
        ALTER TABLE IF EXISTS public.models
            ADD CONSTRAINT fk_model_type_id FOREIGN KEY (model_type_id) REFERENCES public.model_type (id) ON DELETE SET NULL NOT VALID;

        ALTER TABLE public.models VALIDATE CONSTRAINT fk_model_type_id;

        -- PREDICTIONS
        ALTER TABLE IF EXISTS public.predictions
            ADD CONSTRAINT fk_model_id FOREIGN KEY (model_id) REFERENCES public.models (id) ON DELETE SET NULL NOT VALID;

        ALTER TABLE public.predictions VALIDATE CONSTRAINT fk_model_id;

        ALTER TABLE IF EXISTS public.predictions
            ADD CONSTRAINT fk_task_id FOREIGN KEY (task_id) REFERENCES public.task (id) ON DELETE CASCADE NOT VALID;

        ALTER TABLE public.predictions VALIDATE CONSTRAINT fk_task_id;

        -- DATASET
        ALTER TABLE IF EXISTS public.dataset
            ADD CONSTRAINT fk_filetype_id FOREIGN KEY (filetype_id) REFERENCES public.filetype (id) ON DELETE SET NULL NOT VALID;

        ALTER TABLE public.dataset VALIDATE CONSTRAINT fk_filetype_id;

        --TASK
        -- dataset id
        ALTER TABLE IF EXISTS public.task
            ADD CONSTRAINT fk_dataset_id FOREIGN KEY (dataset_id) REFERENCES public.dataset (id) ON DELETE CASCADE NOT VALID;

        ALTER TABLE public.task VALIDATE CONSTRAINT fk_dataset_id;

        -- project id
        ALTER TABLE IF EXISTS public.task
            ADD CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE NOT VALID;

        ALTER TABLE public.task VALIDATE CONSTRAINT fk_project_id;

        -- annotation id
        ALTER TABLE IF EXISTS public.task
            ADD CONSTRAINT fk_annotation_id FOREIGN KEY (annotation_id) REFERENCES public.annotations (id) ON DELETE SET NULL NOT VALID;

        ALTER TABLE public.task VALIDATE CONSTRAINT fk_annotation_id;

        -- prediction id
        ALTER TABLE IF EXISTS public.task
            ADD CONSTRAINT fk_prediction_id FOREIGN KEY (prediction_id) REFERENCES public.predictions (id) ON DELETE SET NULL NOT VALID;

        ALTER TABLE public.task VALIDATE CONSTRAINT fk_prediction_id;

        -- ANNOTATIONS
        -- users id
        ALTER TABLE IF EXISTS public.annotations
            ADD CONSTRAINT fk_users_id FOREIGN KEY (users_id) REFERENCES public.users (id) ON DELETE NO ACTION NOT VALID;

        ALTER TABLE public.annotations VALIDATE CONSTRAINT fk_users_id;

        -- project id
        ALTER TABLE IF EXISTS public.annotations
            ADD CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE NOT VALID;

        ALTER TABLE public.annotations VALIDATE CONSTRAINT fk_project_id;

        -- task id
        ALTER TABLE IF EXISTS public.annotations
            ADD CONSTRAINT fk_task_id FOREIGN KEY (task_id) REFERENCES public.task (id) ON DELETE CASCADE NOT VALID;

        ALTER TABLE public.annotations VALIDATE CONSTRAINT fk_task_id;

        -- PROJECT_DATASET (Many-to-Many)
        -- project id
        ALTER TABLE IF EXISTS public.project_dataset
            ADD CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE NOT VALID;

        ALTER TABLE public.project_dataset VALIDATE CONSTRAINT fk_project_id;

        -- dataset id
        ALTER TABLE IF EXISTS public.project_dataset
            ADD CONSTRAINT fk_dataset_id FOREIGN KEY (dataset_id) REFERENCES public.dataset (id) ON DELETE CASCADE NOT VALID;

        ALTER TABLE public.project_dataset VALIDATE CONSTRAINT fk_dataset_id;

        -- EDITOR
        -- project id
        ALTER TABLE IF EXISTS public.editor
            ADD CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE NOT VALID;

        ALTER TABLE public.editor VALIDATE CONSTRAINT fk_project_id;

        -- PROJECT_TRAINING (Many-to-Many)
        -- project id
        ALTER TABLE IF EXISTS public.project_training
            ADD CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE NOT VALID;

        ALTER TABLE public.project_training VALIDATE CONSTRAINT fk_project_id;

        -- training id
        ALTER TABLE IF EXISTS public.project_training
            ADD CONSTRAINT fk_training_id FOREIGN KEY (training_id) REFERENCES public.training (id) ON DELETE CASCADE NOT VALID;

        ALTER TABLE public.project_training VALIDATE CONSTRAINT fk_training_id;

        -- TRAINING_DATASET (Many-to-Many)
        -- training id
        ALTER TABLE IF EXISTS public.training_dataset
            ADD CONSTRAINT fk_training_id FOREIGN KEY (training_id) REFERENCES public.training (id) ON DELETE CASCADE NOT VALID;

        ALTER TABLE public.training_dataset VALIDATE CONSTRAINT fk_training_id;

        -- dataset id
        ALTER TABLE IF EXISTS public.training_dataset
            ADD CONSTRAINT fk_dataset_id FOREIGN KEY (dataset_id) REFERENCES public.dataset (id) ON DELETE CASCADE NOT VALID;

        ALTER TABLE public.training_dataset VALIDATE CONSTRAINT fk_dataset_id;

        END;
            """
    initialise_database_vars = [f"{argon2.hash('admin')}"]
    try:
        logger.info("Creating relations in database.......")
        db_no_fetch(initialise_database_SQL, conn, initialise_database_vars)
        logger.info("Successfully created relations in database.......")

    except psycopg2.Error as e:
        logger.error(e)


def initialise_database_pipeline(conn, dsn: dict) -> DatabaseStatus:
    """Pipeliine to Initialise Database for the platform

    Args:
        conn (psycopg2.connection): Connection object for PostgreSQL
    """

    # try:
    # check if database exists
    if not check_if_database_exist(datname=APP_DATABASE_NAME,
                                    conn=conn):

        # if not,create database with the name
        create_database(database_name=APP_DATABASE_NAME,
                        conn=conn)
        # closing the old connection to the existing database, before creating
        # a new connection to the new database
        conn.close()
    else:
        logger.info(f"Database '{APP_DATABASE_NAME}' already exists")

    # create new DSN for the new database name
    dsn['dbname'] = APP_DATABASE_NAME
    # NOTE: this is the FIRST ever connection to the new database
    # not using init_connection as this decorated function would cache even on error
    # conn = init_connection(**dsn)
    conn = connect_db(**dsn)

    if not check_if_table_exist('project', conn=conn):
        # then create relation in the database
        logger.info('Creating relation database ...')
        create_relation_database(conn=conn)
    else:
        logger.info(
            f"Tables already exist in database '{APP_DATABASE_NAME}'")

    # also scrape model details online and setup the `models` table if not exists
    if not check_if_pretrained_models_exist(conn):
        logger.info("Scraping all details of pretrained models")
        scrape_setup_model_details(conn)
    else:
        logger.info("Pretrained model data already exists in database")

        return DatabaseStatus.Exist
    # except Exception as e:
    #     logger.error(e)
    #     return DatabaseStatus.NotExist
