-- Headline KPIs for the latest quarter vs the prior quarter.
-- Business question: "How did we do this quarter, and in which direction?"
WITH booked AS (
  SELECT
    CASE
      WHEN o.order_date BETWEEN '2026-07-01' AND '2026-09-30' THEN 'current'
      WHEN o.order_date BETWEEN '2026-04-01' AND '2026-06-30' THEN 'prior'
    END AS period,
    o.order_id,
    o.customer_id,
    o.quantity * o.unit_price * (1 - o.discount) AS net_revenue,
    o.quantity * p.cost_price AS cogs,
    o.discount
  FROM orders o
  JOIN products p ON p.product_id = o.product_id
  WHERE o.status NOT IN ('Cancelled', 'Returned')
    AND o.unit_price IS NOT NULL
)
SELECT
  period,
  ROUND(SUM(net_revenue), 2) AS revenue,
  COUNT(DISTINCT order_id) AS orders,
  COUNT(DISTINCT customer_id) AS active_customers,
  ROUND(SUM(net_revenue) / NULLIF(COUNT(DISTINCT order_id), 0), 2) AS avg_order_value,
  ROUND(100.0 * SUM(net_revenue - cogs) / NULLIF(SUM(net_revenue), 0), 1) AS gross_margin_pct,
  ROUND(100.0 * AVG(discount), 1) AS avg_discount_pct
FROM booked
WHERE period IS NOT NULL
GROUP BY period
ORDER BY period;
