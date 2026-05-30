<?php

namespace Pankaj\MHFSafeguard\XF\Service\Thread;

use Pankaj\MHFSafeguard\XF\Service\Post\Preparer;

class Creator extends XFCP_Creator
{
    public function checkForSpam()
    {
        parent::checkForSpam();

        if ($this->thread->discussion_state === 'visible')
        {
            /** @var Preparer $preparer */
            $preparer = $this->postPreparer;
            $preparer->runMhfSafeguardScan();
        }
    }
}
