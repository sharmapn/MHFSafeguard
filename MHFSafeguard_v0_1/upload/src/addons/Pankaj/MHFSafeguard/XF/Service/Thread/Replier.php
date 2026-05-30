<?php

namespace Pankaj\MHFSafeguard\XF\Service\Thread;

use Pankaj\MHFSafeguard\XF\Service\Post\Preparer;

class Replier extends XFCP_Replier
{
    public function checkForSpam()
    {
        parent::checkForSpam();

        if ($this->post->message_state === 'visible')
        {
            /** @var Preparer $preparer */
            $preparer = $this->postPreparer;
            $preparer->runMhfSafeguardScan();
        }
    }
}
