# You may also refer to the notebook in the same directory as this script for reference

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import pandas as pd
import psycopg2
import requests
import toml
from bs4 import BeautifulSoup
from core.utils.log import logger
from path_desc import (
    CLASSIF_MODELS_NAME_PATH,
    SEGMENT_MODELS_TABLE_PATH,
    TFOD_MODELS_TABLE_PATH,
)
from psycopg2.extras import DictCursor, NamedTupleCursor

SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib


def connect_db(**context):
    # load from toml file if context is not provided
    if not context:
        # assuming you run this script from the root directory of this project
        toml_filepath = ".streamlit/secrets.toml"
        parsed = toml.load(open(toml_filepath))

        # # Connect to PostgreSQL db
        conn = psycopg2.connect(**parsed['postgres'])
    else:
        conn = psycopg2.connect(**context)

    # Check and verify connection
    with conn.cursor(cursor_factory=NamedTupleCursor) as cur:
        cur.execute('SELECT version();')
        conn.commit()

        # display the PostgreSQL database server version
        db_version = cur.fetchone().version

    print(f"PostgreSQL database version: {db_version}")
    print(f"PostgreSQL connection status: {conn.info.status}")
    print(
        f"You are connected to database '{conn.info.dbname}' as user '{conn.info.user}' on host '{conn.info.host}' at port '{conn.info.port}'.")
    return conn

    # ************************** DB Functions **************************


def db_fetchone(sql_message, query_vars=None, conn=None, return_output=False, fetch_col_name=False, return_dict=False):
    assert conn is not None, "Please connect to db and pass in for faster computation."

    with conn.cursor(cursor_factory=NamedTupleCursor) as cur:
        try:
            if query_vars:
                cur.execute(sql_message, query_vars)
            else:
                cur.execute(sql_message)

            conn.commit()
            if not return_output:
                return
            return_one = cur.fetchone()  # return tuple

            if return_dict:
                # Convert results to pure Python dictionary
                if return_one is not None:
                    return_one = dict(return_one)

            if fetch_col_name:
                # Obtain Column names from query
                column_names = [desc[0] for desc in cur.description]
                return return_one, column_names
            else:
                return return_one
        except psycopg2.Error as e:
            conn.rollback()
            print(e)


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
                    return_all = [dict(row) for row in return_all]

                if fetch_col_name:
                    column_names = [desc[0] for desc in cur.description]
                    return return_all, column_names
                else:
                    return return_all
            except psycopg2.Error as e:
                logger.error(e)


def check_if_pretrained_models_exist(conn):
    """Check whether the CSV files exists, and also whether data is stored in database"""
    sql_query = """
        SELECT id FROM models WHERE model_type_id = 1;
    """
    record = db_fetchone(sql_query, conn=conn, return_output=True)
    if TFOD_MODELS_TABLE_PATH.exists() and \
            CLASSIF_MODELS_NAME_PATH.exists() and \
            SEGMENT_MODELS_TABLE_PATH.exists() and \
            record:
        return True
    return False


def delete_pt_model_db_data(conn):
    """
    Delete existing pretrained models before inserting `model_type_id` = 1 for
    pretrained_models type

    NOTE: This is actually only needed for debugging, real installation only needs
    to run this once, hence does not require deleting previously stored pretrained_models
    """
    sql_query = """
    DELETE FROM public.models WHERE model_type_id = 1
    """
    db_fetchone(sql_query, conn=conn)

    # restart the sequence so that the primary key `id` would
    # start at 1 for pretrained models
    # NOTE: will change this to start with the max ID later
    table_name = 'models'
    sequence_name = f'{table_name}_id_seq'
    sql_query = f"""
        SELECT SETVAL('{sequence_name}', 1, false);
    """
    db_fetchone(sql_query, conn=conn)


def query_existing_pt_model_names(conn) -> List[str]:
    sql_query = """
        SELECT name FROM models WHERE model_type_id = 1;
    """
    records = db_fetchall(sql_query, conn=conn)
    existing_models = [r.name for r in records]
    return existing_models


def save_df_if_not_exists(
        new_df: pd.DataFrame, existing_csv_path: Path) -> bool:
    """Save the CSV file if it does not exist, or if the existing CSV data is not
    equal to the newly scraped DataFrame.

    i.e. Skip saving if the CSV file already exists and is the same as the new data"""
    exists = False
    if existing_csv_path.exists():
        existing_df = pd.read_csv(existing_csv_path)
        is_equal = existing_df.equals(new_df)
        logger.debug(f"{is_equal = }")
        # exists = True if new data is equal to existing data
        exists = is_equal

    if not exists:
        logger.info("Saving CSV file")
        new_df.to_csv(existing_csv_path, index=False)


def get_new_model_df(
        new_df: pd.DataFrame, existing_model_names: List[str]) -> pd.DataFrame:
    """Get the DataFrame rows for which the models do not exist in the database"""
    model_col_name = new_df.columns[0]
    new_model_names = set(new_df[model_col_name]).difference(
        existing_model_names)
    new_model_df = new_df[new_df[model_col_name].isin(new_model_names)]
    return new_model_df


def insert_to_db_if_not_exists(new_df: pd.DataFrame, conn, existing_model_names: List[str]):
    if existing_model_names:
        new_model_df = get_new_model_df(new_df, existing_model_names)
        if new_model_df.empty:
            logger.info(
                "The scraped pretrained model data already exists in the database")
            return
        df = new_model_df
    else:
        df = new_df

    sql_query = """
                INSERT INTO public.models (
                    name,
                    metrics,
                    model_type_id,
                    framework_id,
                    deployment_id
                    )
                OVERRIDING SYSTEM VALUE
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                    )
    """
    logger.info("Inserting pretrained model data into database")
    for row in df.values:
        query_vars = row.tolist()
        db_fetchone(sql_query, query_vars, conn=conn)


def scrape_tfod_data(conn, existing_model_names: List[str]):
    import re

    logger.info("Scraping TensorFlow Models information")
    link = "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/g3doc/tf2_detection_zoo.md"
    response = requests.get(link)
    markdown_content = response.text
    table_pattern = re.compile(
        r'\[(.+)\]\((http:.+)\).*\|\s*(\S+)\s*\|\s*(\S+\.*\S*)\s*\|\s*(\S+)')
    matches = table_pattern.findall(markdown_content)
    df = pd.DataFrame(
        matches,
        columns=['Model Name', 'model_links', 'Speed (ms)',
                 'COCO mAP', 'Outputs'])
    df = df[["Model Name", "Speed (ms)", "COCO mAP",
             "Outputs", "model_links"]]

    # save to CSV before formatting for database
    save_df_if_not_exists(df, TFOD_MODELS_TABLE_PATH)

    # Modifying to insert into database
    # Need columns: id, name, description, metrics, model_path, model_type_id,
    #  framework_id, deployment_id
    new_df = df[['Model Name']].copy()
    new_df['metrics'] = json.dumps({'metrics': 'mAP'})
    new_df['model_type_id'] = 1  # for pretrained_model type
    new_df['framework_id'] = 1  # for Tensorflow type
    new_df['deployment_id'] = 2  # for Object Detection

    insert_to_db_if_not_exists(new_df, conn, existing_model_names)


def scrape_classif_model_data(conn, existing_model_names: List[str]):
    logger.info("Scraping Keras Pretrained Classification Models information")
    URL = "https://www.tensorflow.org/api_docs/python/tf/keras/applications"
    data = requests.get(URL)
    soup = BeautifulSoup(data.text, 'html.parser')

    names = soup.select("code")
    model_names = []
    # NOTE: This might not work in the future depending on how TensorFlow
    # structure their documentation
    for i in names:
        if '(...)' in i.text:
            link = i.text.replace('(...)', '')
            model_names.append(link)

    df = pd.DataFrame(model_names, columns=['Model Name'])
    # save to CSV before formatting for database
    save_df_if_not_exists(df, CLASSIF_MODELS_NAME_PATH)

    df['metrics'] = json.dumps({"metrics": ["Accuracy", "F1 Score"]})
    df['model_type_id'] = 1  # for pretrained_model type
    df['framework_id'] = 1  # for Tensorflow type
    df['deployment_id'] = 1  # for image classification

    insert_to_db_if_not_exists(df, conn, existing_model_names)


def scrape_keras_unet_data(conn, existing_model_names: List[str]):
    logger.info("Scraping Keras U-net Models information")
    URL = "https://github.com/yingkaisha/keras-unet-collection"
    data = requests.get(URL)
    soup = BeautifulSoup(data.text, 'html.parser')

    tables = soup.select("table")
    for table in tables:
        if 'models' in table.th.text:
            models_table = table
        if 'losses' in table.th.text:
            losses_table = table

    model_links = models_table.select("td a")
    model_links = [i['href'] for i in model_links]

    model_df = pd.read_html(models_table.prettify())[0]
    model_df['links'] = model_links
    model_df.columns = ["model_func", "Model Name", "Reference", "links"]
    # dropping the last two models built with Transformers as they don't work for new NumPy version,
    # refer to the repo for more details https://github.com/yingkaisha/keras-unet-collection
    model_df.drop(model_df.index[-2:], inplace=True)
    save_df_if_not_exists(model_df, SEGMENT_MODELS_TABLE_PATH)

    # # for the available loss functions
    loss_df = pd.read_html(losses_table.prettify())[0]

    # dropping experimental loss functions
    experimental_row_idxs = loss_df.loc[loss_df['Name'].str.contains(
        'experimental')].index
    loss_df = loss_df.drop(experimental_row_idxs).reset_index(drop=True)

    loss_links = losses_table.select("td a")
    loss_links = [i['href'] for i in loss_links]

    loss_df['links'] = loss_links
    loss_df.columns = ['loss_func', 'Name', 'Reference', 'links']

    df = model_df[['Model Name']].copy()
    df['metrics'] = json.dumps(
        {"metrics": "mAP", "losses": loss_df.loss_func.tolist()})
    df['model_type_id'] = 1  # for pretrained_model type
    df['framework_id'] = 1  # for Tensorflow type
    df['deployment_id'] = 3  # for image segmentation

    insert_to_db_if_not_exists(df, conn, existing_model_names)


def update_model_sequence(conn):
    """
    Update the ID sequence to start from latest ID + 1
    use this or use ALTER SEQUENCE. False to ensure to take next value
    https://coderedirect.com/questions/335605/postgresql-using-subqueries-with-alter-sequence-expressions
    """
    table_name = 'models'
    sequence_name = f'{table_name}_id_seq'
    sql_msg = f"""
        SELECT SETVAL('{sequence_name}', (SELECT MAX(id) + 1 FROM {table_name}), false)
    """
    db_fetchone(sql_msg, conn=conn)


def scrape_setup_model_details(conn):
    existing_model_names = query_existing_pt_model_names(conn)

    scrape_tfod_data(conn, existing_model_names)

    scrape_classif_model_data(conn, existing_model_names)

    scrape_keras_unet_data(conn, existing_model_names)

    update_model_sequence(conn)


def main():
    # connect to DB using .streamlit/secrets.toml
    conn = connect_db()
    # then scrape all the pretrained model details and update our DB
    scrape_setup_model_details(conn)


if __name__ == '__main__':
    main()
