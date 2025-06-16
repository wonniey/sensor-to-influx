from influxdb_client import InfluxDBClient

client = InfluxDBClient(
    url="https://your-influx-url",
    token="your-token",
    org="your-org"
)

query = '''
from(bucket: "your-bucket")
  |> range(start: -1h)
  |> filter(fn: (r) => r["_measurement"] == "fan_predictions")
'''

df = client.query_api().query_data_frame(query=query)
print(df[["_time", "_value", "source"]])
