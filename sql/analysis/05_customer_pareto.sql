-- Revenue concentration: what share of revenue comes from the top customers?
-- Techniques: ROW_NUMBER, window totals, rank-based bucketing.
WITH customer_revenue AS (
  SELECT
    o.customer_id,
    SUM(o.quantity * o.unit_price * (1 - o.discount)) AS revenue
  FROM orders o
  WHERE o.status NOT IN ('Cancelled', 'Returned')
    AND o.unit_price IS NOT NULL
  GROUP BY o.customer_id
),
ranked AS (
  SELECT
    customer_id,
    revenue,
    ROW_NUMBER() OVER (ORDER BY revenue DESC) AS revenue_rank,
    COUNT(*) OVER () AS customers,
    SUM(revenue) OVER () AS total_revenue
  FROM customer_revenue
)
SELECT
  CASE
    WHEN revenue_rank <= 0.1 * customers THEN 'Top 10%'
    WHEN revenue_rank <= 0.2 * customers THEN 'Next 10%'
    WHEN revenue_rank <= 0.5 * customers THEN 'Middle 30%'
    ELSE 'Bottom 50%'
  END AS customer_bucket,
  COUNT(*) AS customers,
  ROUND(SUM(revenue), 2) AS revenue,
  ROUND(100.0 * SUM(revenue) / MAX(total_revenue), 1) AS revenue_share_pct
FROM ranked
GROUP BY customer_bucket
ORDER BY MIN(revenue_rank);
