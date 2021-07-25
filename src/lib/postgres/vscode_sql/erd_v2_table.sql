-- USERS table ------------------------------------------------
CREATE TABLE IF NOT EXISTS public.users (
    id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    emp_id text UNIQUE,
    username text NOT NULL UNIQUE,
    first_name text,
    last_name text,
    email text,
    department text,
    position text,
    psd text NOT NULL,
    roles_id integer NOT NULL,
    account_status character varying(30) NOT NULL CHECK (account_status IN ('NEW', 'ACTIVE', 'LOCKED', 'LOGGED_IN', 'LOGGED_OUT')),
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_activity timestamp with time zone,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

CREATE TRIGGER users_update
    BEFORE UPDATE ON public.users
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.users OWNER TO shrdc;

--  ROLES table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.roles (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name character varying(50) NOT NULL,
    page_access_list text[],
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

ALTER TABLE public.roles OWNER TO shrdc;

CREATE TRIGGER roles_update
    BEFORE UPDATE ON public.roles
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

-- INSERT pre-defined values into ROLES
INSERT INTO public.roles (
    name,
    page_access_list)
VALUES (
    'Administrator',
    ARRAY['Login', 'Project', 'Dataset', 'Editor', 'Model Training', 'Deployment', 'User Management']),
(
    'Developer 1 (Deployment)',
    ARRAY['Login', 'Project', 'Dataset', 'Editor', 'Model Training', 'Deployment']),
(
    'Developer 2 (Model Training)',
    ARRAY['Login', 'Project', 'Dataset', 'Editor', 'Model Training']),
(
    'Annotator',
    ARRAY['Login', 'Project', 'Dataset', 'Editor']);

--  SESSION_LOG table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.session_log (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (CYCLE INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    users_id bigint NOT NULL,
    login_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    logout_at timestamp with time zone NOT NULL,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

ALTER TABLE public.session_log OWNER TO shrdc;

--  PROJECT table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.project (
    id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name text NOT NULL UNIQUE,
    description text,
    project_path text NOT NULL,
    deployment_id integer,
    training_id bigint,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

CREATE TRIGGER project_update
    BEFORE UPDATE ON public.project
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.project OWNER TO shrdc;

--  TRAINING table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.training (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name text NOT NULL UNIQUE,
    description text,
    training_param jsonb[],
    augmentation jsonb[],
    model_id bigint,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    project_id bigint NOT NULL,
    pre_trained_model_id bigint,
    framework_id bigint,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

CREATE TRIGGER training_update
    BEFORE UPDATE ON public.training
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.training OWNER TO shrdc;

-- TRAINING_LOG table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.training_log (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (CYCLE INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    users_id bigint NOT NULL,
    training_id bigint NOT NULL,
    start_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP, --update using Python
    end_at timestamp with time zone NOT NULL,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

ALTER TABLE public.training_log OWNER TO shrdc;

-- PRE-TRAINED MODELS table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.pre_trained_models (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (CYCLE INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name text NOT NULL,
    model_path text,
    framework_id bigint,
    deployment_id integer,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

CREATE TRIGGER pre_trained_models_update
    BEFORE UPDATE ON public.pre_trained_models
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.pre_trained_models OWNER TO shrdc;

INSERT INTO public.pre_trained_models (
    name,
    framework_id,
    model_path,
    deployment_id)
VALUES (
    '[TF] SSD MobileNet V2 FPNLite 320x320',
    (
        SELECT
            f.id
        FROM
            public.framework f
        WHERE
            f.name = 'TensorFlow'), './pre-trained-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 2), ('[TF] SSD ResNet50 V1 FPN 640x640 (RetinaNet50)', (
        SELECT
            f.id
        FROM
            public.framework f
        WHERE
            f.name = 'TensorFlow'), './pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8', 2);

-- MODELS table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.models (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    model_name text NOT NULL UNIQUE,
    model_path text,
    training_id bigint,
    framework_id bigint,
    deployment_id integer,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

CREATE TRIGGER models_update
    BEFORE UPDATE ON public.models
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.models OWNER TO shrdc;

-- PREDICTIONS table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.predictions (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    results jsonb[],
    task_id bigint NOT NULL,
    model_id bigint,
    pre_trained_model_id bigint,
    score double precision,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

CREATE TRIGGER predictions_update
    BEFORE UPDATE ON public.predictions
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.predictions OWNER TO shrdc;

-- DATASET table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.dataset (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name text NOT NULL UNIQUE,
    description text,
    file_type character varying(100) NOT NULL,
    dataset_path text NOT NULL,
    dataset_size integer,
    deployment_id integer,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

CREATE TRIGGER dataset_update
    BEFORE UPDATE ON public.dataset
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.dataset OWNER TO shrdc;

--  DEPLOYMENT_TYPE table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.deployment_type (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name character varying(100) NOT NULL,
    template text,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

ALTER TABLE public.deployment_type OWNER TO shrdc;

INSERT INTO public.deployment_type (
    name)
VALUES (
    'Image Classification'),
(
    'Object Detection with Bounding Boxes'),
(
    'Semantic Segmentation with Polygons'),
(
    'Semantic Segmentation with Masks');

--  TASK table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.task (
    id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name text NOT NULL,
    dataset_id bigint NOT NULL,
    project_id bigint,
    annotation_id bigint,
    prediction_id bigint,
    is_labelled boolean NOT NULL DEFAULT FALSE,
    skipped boolean NOT NULL DEFAULT FALSE,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

CREATE TRIGGER task_update
    BEFORE UPDATE ON public.task
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.task OWNER TO shrdc;

-- ANNOTATIONS table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.annotations (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    results jsonb[],
    annotation_type_id integer,
    project_id bigint NOT NULL,
    users_id bigint NOT NULL,
    task_id bigint NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

CREATE TRIGGER annotations_update
    BEFORE UPDATE ON public.annotations
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.annotations OWNER TO shrdc;

-- ANNOTATION_TYPE table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.annotation_type (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name character varying(100) NOT NULL,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

ALTER TABLE public.annotation_type OWNER TO shrdc;

-- PROJECT_DATASET table (Many-to-Many) --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.project_dataset (
    project_id bigint NOT NULL,
    dataset_id bigint NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (project_id, dataset_id))
TABLESPACE image_labelling;

CREATE TRIGGER project_dataset_update
    BEFORE UPDATE ON public.project_dataset
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.project_dataset OWNER TO shrdc;

-- EDITOR table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.editor (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name character varying(50) NOT NULL,
    editor_config text,
    labels text[],
    project_id bigint NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

CREATE TRIGGER editor_update
    BEFORE UPDATE ON public.editor
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.editor OWNER TO shrdc;

-- FRAMEWORK table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.framework (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name character varying(50) NOT NULL,
    link text,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

CREATE TRIGGER framework_update
    BEFORE UPDATE ON public.framework
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.framework OWNER TO shrdc;

INSERT INTO public.framework (
    name,
    link)
VALUES (
    'TensorFlow',
    'https://www.tensorflow.org/'),
(
    'PyTorch',
    'https://pytorch.org/'),
(
    'Scikit-learn',
    'https://scikit-learn.org/stable/'),
(
    'Caffe',
    'https://caffe.berkeleyvision.org/'),
(
    'MXNet',
    'https://mxnet.apache.org/'),
(
    'ONNX',
    'https://onnx.ai/');

-- PROJECT_TRAINING table (Many-to-Many) ---------------------------
CREATE TABLE IF NOT EXISTS public.project_training (
    project_id bigint NOT NULL,
    training_id bigint NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (project_id, training_id))
TABLESPACE image_labelling;

CREATE TRIGGER project_training_update
    BEFORE UPDATE ON public.project_training
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.project_training OWNER TO shrdc;

-- TRAINING_DATASET table (Many-to-Many) ---------------------------
CREATE TABLE IF NOT EXISTS public.training_dataset (
    training_id bigint NOT NULL,
    dataset_id bigint NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (training_id, dataset_id) TABLESPACE image_labelling;

CREATE TRIGGER training_dataset_update
    BEFORE UPDATE ON public.training_dataset
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.training_dataset OWNER TO shrdc;

