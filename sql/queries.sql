
-- INSTAMART CUSTOMER INTELLIGENCE
-- Seven SQL business analysis queries

-- 1. Total number of customers
SELECT COUNT(*) AS total_customers
FROM customers;

-- 2. Total orders and completed orders
SELECT
    COUNT(*) AS total_orders,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)
        AS completed_orders
FROM orders;

-- 3. Revenue by product category
SELECT
    p.category,
    ROUND(SUM(o.order_amount), 2) AS total_revenue
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.category
ORDER BY total_revenue DESC;

-- 4. Monthly revenue
SELECT
    strftime('%Y-%m', order_ts) AS order_month,
    ROUND(SUM(order_amount), 2) AS monthly_revenue
FROM orders
WHERE status = 'completed'
GROUP BY order_month
ORDER BY order_month;

-- 5. Top 10 products by revenue
SELECT
    p.product_name,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS total_revenue
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.product_id, p.product_name, p.category
ORDER BY total_revenue DESC
LIMIT 10;

-- 6. Revenue by customer city
SELECT
    c.city,
    ROUND(SUM(o.order_amount), 2) AS total_revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY c.city
ORDER BY total_revenue DESC;

-- 7. Top 10 customers by completed order value
SELECT
    c.customer_id,
    c.city,
    COUNT(o.order_id) AS completed_orders,
    ROUND(SUM(o.order_amount), 2) AS total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
GROUP BY c.customer_id, c.city
ORDER BY total_spent DESC
LIMIT 10;
