
import duckdb
import os

def main():
    con = duckdb.connect()

    # Enable DuckDB's built-in HTTP server extension
    con.install_extension("httpserver", repository="community")
    con.load_extension("httpserver")

    # Parquet file as a table
    con.execute("CREATE TABLE wildfire AS SELECT * FROM '/data/wildfire_data.parquet';")

    # Start  DuckDB HTTP server
    os.environ['DUCKDB_HTTPSERVER_FOREGROUND'] = '1'
    info = con.execute("SELECT httpserve_start('0.0.0.0', 9999, '');").fetchall()
    print(info[0])
    

if __name__ == "__main__":
    main()

