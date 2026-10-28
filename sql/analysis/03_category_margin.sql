-- Revenue, gross profit, margin and revenue share by category (latest quarter).
-- Business question: "Where do we actually make money?"
WITH cat AS (
  SELECT
    p.category,
    SUM(o.quantity * o.unit_price * (1 - o.discount)) AS revenue,
    SUM(o.quantity * o.unit_price * (1 - o.discount) - o.quantity * p.cost_price) AS gross_profit
  FROM orders o
  JOIN products p ON p.product_id = o.product_id
  WHERE o.status NOT IN ('Cancelled', 'Returned')
    AND o.unit_price IS NOT NULL
    AND o.order_date BETWEEN '2026-07-01' AND '2026-09-30'
  GROUP BY p.category
)
SELECT
  category,
  ROUND(revenue, 2) AS revenue,
  ROUND(gross_profit, 2) AS gross_profit,
  ROUND(100.0 * gross_profit / NULLIF(revenue, 0), 1) AS gross_margin_pct,
  ROUND(100.0 * revenue / SUM(revenue) OVER (), 1) AS revenue_share_pct
FROM cat
ORDER BY revenue DESC;
