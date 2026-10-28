-- Monthly acquisition cohorts: share of each cohort that ordered again later.
-- Cohort = month of the first booked order for each customer.
WITH booked AS (
  SELECT DISTINCT o.customer_id, substr(o.order_date, 1, 7) AS order_month
  FROM orders o
  WHERE o.status NOT IN ('Cancelled', 'Returned')
),
first_order AS (
  SELECT customer_id, MIN(order_month) AS cohort_month
  FROM booked
  GROUP BY customer_id
),
activity AS (
  SELECT
    f.cohort_month,
    b.customer_id,
    (CAST(substr(b.order_month, 1, 4) AS INTEGER) * 12 + CAST(substr(b.order_month, 6, 2) AS INTEGER))
    - (CAST(substr(f.cohort_month, 1, 4) AS INTEGER) * 12 + CAST(substr(f.cohort_month, 6, 2) AS INTEGER))
      AS months_since_first
  FROM booked b
  JOIN first_order f ON f.customer_id = b.customer_id
),
cohort_size AS (
  SELECT cohort_month, COUNT(*) AS customers FROM first_order GROUP BY cohort_month
)
SELECT
  a.cohort_month,
  s.customers AS cohort_customers,
  ROUND(100.0 * COUNT(DISTINCT CASE WHEN a.months_since_first BETWEEN 1 AND 3 THEN a.customer_id END)
        / s.customers, 1) AS repeat_within_3m_pct,
  ROUND(100.0 * COUNT(DISTINCT CASE WHEN a.months_since_first >= 1 THEN a.customer_id END)
        / s.customers, 1) AS repeat_ever_pct
FROM activity a
JOIN cohort_size s ON s.cohort_month = a.cohort_month
GROUP BY a.cohort_month, s.customers
ORDER BY a.cohort_month;
