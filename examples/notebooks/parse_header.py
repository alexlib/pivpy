import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import re
    from typing import Dict, Tuple

    return Dict, Tuple, re


@app.cell
def _(Dict, re):
    def datasetauxdata_to_dict(text: str) -> Dict:
        result = re.findall(r"DATASETAUXDATA \w+=\"[\w\.]+\"", text)
        dict = {}

        for item in result:
            pair = item.replace("DATASETAUXDATA ", "")
            key_value = pair.split("=")
            dict[key_value[0]] = key_value[1].replace("\"", "")

        # print(dict)
        return dict

    return (datasetauxdata_to_dict,)


@app.cell
def _(datasetauxdata_to_dict):
    text=r"""
    TITLE="D:\Experiments2021\iai_0107_\exp1_\Analysis\exp1_006014.T000.D000.P001.H000.L.vec" VARIABLES="X mm", "Y mm", "U m/s", "V m/s", "CHC", DATASETAUXDATA Application="PIV" DATASETAUXDATA SourceImageWidth="4008" DATASETAUXDATA SourceImageHeight="2672" DATASETAUXDATA MicrometersPerPixelX="85.809998" DATASETAUXDATA MicrometersPerPixelY="85.809998" DATASETAUXDATA LengthUnit="mm" DATASETAUXDATA OriginInImageX="0.000000" DATASETAUXDATA OriginInImageY="0.000000" DATASETAUXDATA MicrosecondsPerDeltaT="50.000000" DATASETAUXDATA TimeUnit="ms" DATASETAUXDATA SecondaryPeakNumber="0" DATASETAUXDATA DewarpedImageSource="0" ZONE I=124, J=82, F=POINT
    """

    datasetauxdata_to_dict(text)
    return


@app.cell
def _():
    import pathlib
    from importlib.resources import files
    file = files('pivpy') / 'data/Insight/Run000001.T000.D000.P000.H001.L.vec'
    return file, pathlib


@app.cell
def _(datasetauxdata_to_dict, file):
    with open(file, "r") as f:
        header = f.readline()

    print(header)

    datasetauxdata_to_dict(header)
    return (header,)


@app.cell
def _(Tuple, pathlib, re):
    def parse_header(filename: pathlib.Path)-> Tuple:
        """
        parse_header ( filename)
        Parses header of the file (.vec) to get the variables (typically X,Y,U,V)
        and units (can be m,mm, pix/DELTA_T or mm/sec, etc.), and the size of the
        Dataset by the number of rows and columns.
        Input:
            filename : complete path of the file to read, pathlib.Path
        Returns:
            variables : list of strings
            units : list of strings
            rows : number of rows of the Dataset
            cols : number of columns of the Dataset
            DELTA_T   : time interval between the two PIV frames in microseconds
        """

        # defaults
        frame = 0

        # split path from the filename
        fname = filename.name
        # get the number in a filename if it's a .vec file from Insight
        if "." in fname[:-4]:  # day2a005003.T000.D000.P003.H001.L.vec
            frame = int(re.findall(r"\d+", fname.split(".")[0])[-1])
        elif "_" in filename[:-4]:
            frame = int(
                re.findall(r"\d+", fname.split("_")[1])[-1]
            )  # exp1_001_b.vec, .txt

        with open(filename,"r") as fid:
            header = fid.readline()

        # if the file does not have a header, can be from OpenPIV or elsewhere
        # return None
        if header[:5] != "TITLE":
            variables = ["x", "y", "u", "v"]
            units = ["pix", "pix", "pix", "pix"]
            rows = None
            cols = None
            dt = 0.0
            return (variables, units, rows, cols, dt, frame)

        header_list = (
            header.replace(",", " ").replace("=", " ").replace('"', " ").split()
        )

        # get variable names, typically X,Y,U,V
        variables = header_list[3:12][::2]

        # get units - this is important if it's mm or m/s
        units = header_list[4:12][::2]

        # get the size of the PIV grid in rows x cols
        rows = int(header_list[-5])
        cols = int(header_list[-3])

        # this is also important to know the time interval, DELTA_T
        ind1 = header.find("MicrosecondsPerDeltaT")
        dt = float(header[ind1:].split('"')[1])

        return (variables, units, rows, cols, dt, frame)

    return (parse_header,)


@app.cell
def _(file, parse_header):
    parse_header(file)
    return


@app.cell
def _(header):
    header
    return


if __name__ == "__main__":
    app.run()
