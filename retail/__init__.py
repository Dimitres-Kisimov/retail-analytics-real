"""retail — end-to-end analytics on the UCI Online Retail II dataset (real data).

Modules:
    ingest    load both xlsx sheets, type-cast, raw-quality report, interim cache
    clean     documented cleaning pipeline (every step logs rows in/out)
    eda       honest EDA figures
    rfm       from-scratch RFM segmentation
    forecast  from-scratch weekly forecasting with rolling-origin CV
    exports   executive PDF + Excel workbook
"""

__version__ = "1.0.0"
