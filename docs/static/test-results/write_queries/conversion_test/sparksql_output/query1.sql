CREATE TABLE temp_table
USING ORC
PARTITIONED BY (
  ds
) AS
WITH ranked_events AS (
  SELECT
    event_id,
    user_id,
    event_type,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_timestamp DESC) AS rn
  FROM events
)
SELECT
  event_id,
  user_id,
  event_type
FROM ranked_events
WHERE
  rn = 1