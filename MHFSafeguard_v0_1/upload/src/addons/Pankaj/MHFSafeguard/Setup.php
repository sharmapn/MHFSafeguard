<?php

namespace Pankaj\MHFSafeguard;

use XF\AddOn\AbstractSetup;
use XF\Db\Schema\Create;

class Setup extends AbstractSetup
{
    public function install(array $stepParams = [])
    {
        $this->schemaManager()->createTable('xf_mhfs_scan', function (Create $table)
        {
            $table->addColumn('scan_id', 'int')->autoIncrement()->primaryKey();
            $table->addColumn('content_type', 'varbinary', 50)->setDefault('post');
            $table->addColumn('content_id', 'int')->setDefault(0);
            $table->addColumn('thread_id', 'int')->setDefault(0);
            $table->addColumn('node_id', 'int')->setDefault(0);
            $table->addColumn('user_id', 'int')->setDefault(0);
            $table->addColumn('username', 'varchar', 100)->setDefault('');
            $table->addColumn('message_hash', 'char', 64)->setDefault('');
            $table->addColumn('message_text', 'mediumtext')->nullable();
            $table->addColumn('risk_level', 'varchar', 30)->setDefault('none');
            $table->addColumn('recommended_action', 'varchar', 30)->setDefault('allow');
            $table->addColumn('final_action', 'varchar', 30)->setDefault('allow');
            $table->addColumn('highest_label', 'varchar', 100)->setDefault('');
            $table->addColumn('highest_score', 'decimal', '8, 4')->setDefault(0);
            $table->addColumn('api_success', 'tinyint', 1)->setDefault(0);
            $table->addColumn('api_status_code', 'int')->setDefault(0);
            $table->addColumn('api_error', 'text')->nullable();
            $table->addColumn('flagged_parts_json', 'mediumtext')->nullable();
            $table->addColumn('api_response_json', 'mediumtext')->nullable();
            $table->addColumn('scan_date', 'int')->setDefault(0);

            $table->addKey(['content_type', 'content_id']);
            $table->addKey(['thread_id']);
            $table->addKey(['user_id']);
            $table->addKey(['risk_level']);
            $table->addKey(['scan_date']);
        });
    }

    public function upgrade(array $stepParams = [])
    {
    }

    public function uninstall(array $stepParams = [])
    {
        $this->schemaManager()->dropTable('xf_mhfs_scan');
    }
}
