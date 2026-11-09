
-- 1. Display total number of customers
SELECT COUNT(*) AS total_customers
FROM customers;

-- 2. Display total number of orders
SELECT COUNT(*) AS total_orders
FROM orders;

-- 3. Display order status
SELECT
    status,
    COUNT(*) AS total_orders
FROM orders
GROUP BY status;

-- 4. Calculate total revenue
SELECT
    ROUND(SUM(order_amount), 2) AS total_revenue
FROM orders;

-- 5. Find average order value
SELECT
    ROUND(AVG(order_amount), 2) AS average_order_value
FROM orders;
