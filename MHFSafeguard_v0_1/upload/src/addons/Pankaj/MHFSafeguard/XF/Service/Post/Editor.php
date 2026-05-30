<?php

namespace Pankaj\MHFSafeguard\XF\Service\Post;

class Editor extends XFCP_Editor
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
