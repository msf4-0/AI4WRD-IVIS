import { ReactElement, useEffect, useState } from "react";
import {
  Streamlit,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib";
import {
  DataGrid,
  GridRowsProp,
  GridColDef,
  GridRowId,
} from "@material-ui/data-grid";
import { makeStyles } from "@material-ui/styles";

//To store object of Streamlit Theme props
interface ComponentTheme {
  primaryColor: string | undefined;
  secondaryBackgroundColor: string | undefined;
  textColor: string | undefined;
  backgroundColor: string | undefined;
  font: string | undefined;
}

function DataTable({ args, theme }: ComponentProps): ReactElement {
  // TODO #28 Get data from Streamlit Python

  //Define Rows and Columns
  const rows: GridRowsProp = args.rows;
  console.log(rows);
  //Define Columns
  const columns: GridColDef[] = args.columns;
  //Get checkbox boolean
  const checkbox: boolean = args.checkbox;
  console.log(checkbox);
  // Declare theme from Streamlit Component
  let myTheme: ComponentTheme = {
    primaryColor: theme?.primaryColor,
    secondaryBackgroundColor: theme?.secondaryBackgroundColor,
    textColor: theme?.textColor,
    backgroundColor: theme?.backgroundColor,
    font: theme?.font,
  };

  function customCheckbox() {
    return {
      "& .MuiCheckbox-root svg": {
        width: 16,
        height: 16,
        backgroundColor: "transparent",
        border: `1px solid ${
          myTheme.backgroundColor === "#ffffff" ? "#d9d9d9" : "#98989A"
        }`,
        borderRadius: 2,
      },
      "& .MuiCheckbox-root svg path": {
        display: "none",
      },
      "& .MuiCheckbox-root.Mui-checked:not(.MuiCheckbox-indeterminate) svg": {
        backgroundColor: "#1890ff",
        borderColor: "#1890ff",
      },
      "& .MuiCheckbox-root.Mui-checked .MuiIconButton-label:after": {
        position: "absolute",
        display: "table",
        border: "2px solid #fff",
        borderTop: 0,
        borderLeft: 0,
        transform: "rotate(45deg) translate(-50%,-50%)",
        opacity: 1,
        transition: "all .2s cubic-bezier(.12,.4,.29,1.46) .1s",
        content: '""',
        top: "50%",
        left: "39%",
        width: 5.71428571,
        height: 9.14285714,
      },
      "& .MuiCheckbox-root.MuiCheckbox-indeterminate .MuiIconButton-label:after":
        {
          width: 8,
          height: 8,
          backgroundColor: "#1890ff",
          transform: "none",
          top: "39%",
          border: 0,
        },
    };
  }

  const useStyles = makeStyles({
    root: {
      //set border colour based on active background color from Streamlit
      //supports Light and Dark mode
      border: `1px solid ${
        myTheme.backgroundColor === "#ffffff" ? "#d9d9d9" : "#98989A"
      }`,

      height: "auto",
      color:
        myTheme.backgroundColor === "#ffffff"
          ? "rgba(0,0,0,1)"
          : "rgba(255,255,255,1)",
      fontFamily: [
        myTheme.font,
        "-apple-system",
        "BlinkMacSystemFont",
        '"Segoe UI"',
        "Roboto",
        '"Helvetica Neue"',
        "Arial",
        "sans-serif",
        '"Apple Color Emoji"',
        '"Segoe UI Emoji"',
        '"Segoe UI Symbol"',
      ].join(","),
      WebkitFontSmoothing: "auto",
      letterSpacing: "normal",
      "& .MuiDataGrid-columnsContainer": {
        backgroundColor:
          myTheme.backgroundColor === "#ffffff"
            ? "#fafafa"
            : myTheme.secondaryBackgroundColor, //working
      },
      "& .MuiDataGrid-iconSeparator": {
        display: "none",
      },
      "& .MuiDataGrid-columnHeader, .MuiDataGrid-cell": {
        //working
        borderRight: `1px solid ${
          myTheme.backgroundColor === "#ffffff"
            ? "#f0f0f0"
            : myTheme.secondaryBackgroundColor
        }`,
      },

      // "& .MuiDataGrid-columnsContainer, .MuiDataGrid-cell": {
      //   borderBottom: `1px solid ${
      //     myTheme.backgroundColor === "#ffffff"
      //       ? "#f0f0f0"
      //       : myTheme.secondaryBackgroundColor
      //   }`,
      // },
      // "& .MuiDataGrid-cell": {
      //   color: myTheme.textColor,
      // },
      "& .MuiPagination-root": {
        color: "secondary",
      },
      ...customCheckbox(), // Custom Checkbox
    },
  });

  // TODO #27 Add dark mode for Pagination
  const classes = useStyles();
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState<number>(10);
  const [selectionModel, setSelectionModel] = useState<GridRowId[]>([]);
  const headerHeight: number = 56;
  const rowHeight: number = 52;
  const footerHeight: number = 55;
  const offset: number = 15;

  console.log(pageSize);

  /* Function to calculate the frame height to render the frame dynamically */
  function frameHeightCalc(size: number) {
    let totalPage = Math.floor(rows.length / size); //calculate the floor divisor to obtain number of pages
    let rows_length = rows.length;
    let frameHeight = 215; /* default height when there is no data */

    //Number of rows at the last page not equal to pageSize
    if (rows_length > 0) {
      if (rows_length % size !== 0 && page === totalPage) {
        //For last page -> frame height is the number of remainder rows
        let remainder = rows.length % size;
        frameHeight = headerHeight + rowHeight * remainder + footerHeight;
      } else {
        frameHeight = headerHeight + rowHeight * size + footerHeight;
      }
    }
    return frameHeight;
  }
  
  const frameHeight = frameHeightCalc(pageSize) + offset;
  console.log("Rows length", rows.length);

  console.log("frame height", frameHeight);
  useEffect(() => Streamlit.setFrameHeight(frameHeight)); // TODO
  /********CALLBACK FUNCTIONS********/

  //Callback to change page
  const onPageChange = (newPage: number) => {
    setPage(newPage);
    Streamlit.setFrameHeight();
  };
  //Callback to modify page size
  const onPageSizeChange = (newPageSize: number) => {
    setPageSize(newPageSize);
    Streamlit.setFrameHeight();
  };
  //Callback when row is selected via Checkbox
  const onSelectionModelChange = (newSelectionModel: GridRowId[]) => {
    setSelectionModel(newSelectionModel);
    Streamlit.setComponentValue(newSelectionModel); //Return Selected Row ID back to Python land
    console.log(newSelectionModel);
  };

  return (
    // <div style={{ height: 500, width: "100%" }}>
    <div style={{ height: 400, width: "100%" }}>
      <div style={{ display: "flex", height: "100%" }}>
        <div style={{ flexGrow: 1 }}>
          <DataGrid
            classes={{ root: classes.root }}
            autoPageSize
            autoHeight={true}
            pagination
            // components={{
            //   Pagination: CustomPagination,
            // }}
            page={page}
            pageSize={pageSize}
            onPageChange={onPageChange}
            onPageSizeChange={onPageSizeChange}
            rows={rows}
            columns={columns}
            rowsPerPageOptions={[5, 10, 20]}
            checkboxSelection={checkbox}
            // disableSelectionOnClick
            onSelectionModelChange={onSelectionModelChange}
            selectionModel={selectionModel}
          />
        </div>
      </div>
    </div>
  );
}

export default withStreamlitConnection(DataTable);
