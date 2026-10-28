-- Month-over-month revenue growth and a 3-month rolling average.
-- Techniques: window functions (LAG, AVG over a row frame).
WITH monthly AS (
  SELECT
    substr(o.order_date, 1, 7) AS month,
    SUM(o.quantity * o.unit_price * (1 - o.discount)) AS revenue,
    COUNT(DISTINCT o.order_id) AS orders
  FROM orders o
  WHERE o.status NOT IN ('Cancelled', 'Returned')
    AND o.unit_price IS NOT NULL
  GROUP BY month
)
SELECT
  month,
  ROUND(revenue, 2) AS revenue,
  orders,
  ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
        / NULLIF(LAG(revenue) OVER (ORDER BY month), 0), 1) AS mom_growth_pct,
  ROUND(AVG(revenue) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2)
    AS rolling_3m_revenue
FROM monthly
ORDER BY month;
