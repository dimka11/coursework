SELECT utc_timestamp ,GB_temperature
FROM weather_data;

SELECT AVG(GB_temperature)
FROM weather_data
WHERE utc_timestamp BETWEEN "1980-01-01T00:00:00Z" AND "1980-01-01T23:00:00Z";

-- sqllite group by window / times group by like
SELECT utc_timestamp, GB_temperature,
avg(GB_temperature) OVER (ORDER BY utc_timestamp ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING) as MovingAverageWindow7
FROM weather_data ORDER BY utc_timestamp;

SELECT utc_timestamp, GB_temperature,
  (SELECT AVG(GB_temperature) FROM weather_data t2
   WHERE datetime(t1.utc_timestamp, '-3 days') <= datetime(t2.utc_timestamp) AND datetime(t1.utc_timestamp, '+3 days') >= datetime(t2.utc_timestamp)
   ) AS MAVG
FROM weather_data t1
GROUP BY strftime('%Y-%m-%d', utc_timestamp);

SELECT utc_timestamp, utc_timestamp - LAG (utc_timestamp, 1, utc_timestamp) OVER (
        ORDER BY utc_timestamp
    ) difference
FROM (
    SELECT utc_timestamp, AVG(GB_temperature) as avgt
    FROM weather_data
    GROUP BY strftime('%Y-%m-%d', utc_timestamp)
    HAVING avgt < -4
);