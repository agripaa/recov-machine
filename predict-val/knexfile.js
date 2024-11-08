module.exports = {
    development: {
      client: 'mysql2',
      connection: {
        host: 'localhost',
        user: 'root',
        password: 'mamank546',
        database: 'fall_detect',
        port: 8889
      },
      pool: { min: 0, max: 7 },
      migrations: {
        tableName: 'knex_migrations'
      }
    },
  };
  