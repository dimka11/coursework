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

select min(date_before), count(utc_timestamp)+1, month, year from
(select strftime('%d',utc_timestamp) - lag(strftime('%d',utc_timestamp)) over (order by utc_timestamp) diff
       , strftime('%d',utc_timestamp) day
       , lag(strftime('%d', utc_timestamp)) over (order by utc_timestamp) day_before
       , strftime('%m', utc_timestamp) month
       ,  lag(strftime('%m', utc_timestamp)) over (order by utc_timestamp) m_before
       , strftime('%Y', utc_timestamp) year
       , lag(strftime('%Y', utc_timestamp)) over (order by utc_timestamp) y_before
       , utc_timestamp
       , lag(utc_timestamp) over (order by utc_timestamp) date_before
from (
     SELECT utc_timestamp, AVG(GB_temperature) as avgt
    FROM weather_data
    GROUP BY strftime('%Y-%m-%d', utc_timestamp)
    HAVING avgt < -2
         )
order by utc_timestamp)
where diff = 1
and month = m_before
and year = y_before
group by  month, year;